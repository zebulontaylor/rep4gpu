use anyhow::{bail, Context, Result};
use bulletproofs::{
    r1cs::{ConstraintSystem, LinearCombination as LC, Prover, R1CSProof, Variable, Verifier},
    BulletproofGens, PedersenGens,
};
use curve25519_dalek::{
    ristretto::{CompressedRistretto, RistrettoPoint},
    scalar::Scalar,
    traits::Identity,
};
use merlin::Transcript;
use rand::rngs::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use std::{fs, path::Path, time::Instant};

const VERSION: &str = "siv-silu-rational34-q24-v1";
const S: i128 = 1 << 24;
const P: i128 = 1 << 30;
const A: [i128; 5] = [0, 134217728, 1108819968, 1612275712, 395427840];
const B: [i128; 5] = [2097152, 62064384, 203923712, 129811904, 8445632];
const GATES_PER_VALUE: usize = 524;

fn scalar(v: i128) -> Scalar {
    let magnitude =
        Scalar::from(u64::try_from(v.unsigned_abs()).expect("circuit integer exceeds u64"));
    if v < 0 {
        -magnitude
    } else {
        magnitude
    }
}
fn range<CS: ConstraintSystem>(cs: &mut CS, v: LC, value: Option<i128>, bits: usize) -> Result<()> {
    let mut sum = LC::default();
    for bit in 0..bits {
        let assignment = value.map(|x| scalar((x >> bit) & 1));
        let (a, b, out) = cs.allocate_multiplier(assignment.map(|x| (x, Scalar::ONE - x)))?;
        cs.constrain(out.into());
        cs.constrain(a + b - Scalar::ONE);
        sum = sum + a * Scalar::from(1u64 << bit);
    }
    cs.constrain(v - sum);
    Ok(())
}
fn nearest<CS: ConstraintSystem>(
    cs: &mut CS,
    raw: LC,
    raw_value: Option<i128>,
    shift: usize,
    quotient_bits: usize,
) -> Result<(Variable, Option<i128>)> {
    let rounded = raw_value.map(|v| v + (1i128 << (shift - 1)));
    let q_value = rounded.map(|v| v >> shift);
    let r_value = rounded.map(|v| v & ((1i128 << shift) - 1));
    let q = cs.allocate(q_value.map(scalar))?;
    let r = cs.allocate(r_value.map(scalar))?;
    range(cs, q.into(), q_value, quotient_bits)?;
    range(cs, r.into(), r_value, shift)?;
    // Bounded quotient and remainder make this an integer floor identity.
    cs.constrain(raw + scalar(1i128 << (shift - 1)) - q * scalar(1i128 << shift) - r);
    Ok((q, q_value))
}
fn polynomial<CS: ConstraintSystem>(
    cs: &mut CS,
    powers: &[LC],
    values: &[Option<i128>],
    coeff: &[i128; 5],
    bits: usize,
) -> Result<(Variable, Option<i128>)> {
    let mut expression = LC::default();
    let mut value = Some(0i128);
    for i in 0..5 {
        expression = expression + powers[i].clone() * scalar(coeff[i]);
        value = value.zip(values[i]).map(|(a, b)| a + coeff[i] * b);
    }
    nearest(cs, expression, value, 30, bits)
}
fn constrain_silu<CS: ConstraintSystem>(
    cs: &mut CS,
    xs: Variable,
    ys: Variable,
    witness: Option<(i128, i128)>,
) -> Result<()> {
    let xv = witness.map(|(x, _)| x);
    let yv = witness.map(|(_, y)| y);
    range(cs, xs.into(), xv.map(|x| x + 16 * S), 30)?;
    range(cs, scalar(32 * S) - xs, xv.map(|x| 16 * S - x), 30)?;
    range(cs, ys.into(), yv.map(|y| y + S), 29)?;
    // Shifted commitments have nonnegative, uniquely bounded integer openings.
    let x: LC = xs - scalar(16 * S);
    let y: LC = ys - scalar(S);
    let (_, _, xx) = cs.multiply(x.clone(), x.clone());
    let (z, zv) = nearest(cs, xx.into(), xv.map(|x| x * x), 26, 31)?;
    let mut powers = vec![LC::from(scalar(P)), z.into()];
    let mut values = vec![Some(P), zv];
    for _ in 0..3 {
        let (_, _, product) = cs.multiply(powers.last().unwrap().clone(), z.into());
        let raw = values.last().copied().unwrap().zip(zv).map(|(a, b)| a * b);
        let (q, v) = nearest(cs, product.into(), raw, 30, 31)?;
        powers.push(q.into());
        values.push(v);
    }
    let (a, av) = polynomial(cs, &powers, &values, &A, 32)?;
    let (b, bv) = polynomial(cs, &powers, &values, &B, 29)?;
    let (_, _, xb) = cs.multiply(x, b.into());
    let (_, _, yd) = cs.multiply(y, b * Scalar::from(2u64));
    let residual: LC = (yd - xb - a * scalar(2 * S)) * Scalar::from(2u64);
    let rv = yv
        .zip(bv)
        .zip(xv)
        .zip(av)
        .map(|(((y, b), x), a)| 2 * (y * 2 * b - x * b - 2 * S * a));
    range(
        cs,
        residual.clone() + b * Scalar::from(2u64),
        rv.zip(bv).map(|(r, b)| r + 2 * b),
        31,
    )?;
    range(
        cs,
        b * Scalar::from(2u64) - residual,
        rv.zip(bv).map(|(r, b)| 2 * b - r),
        31,
    )?;
    Ok(())
}
fn honest_y(x: i128) -> i128 {
    let z = (x * x + (1 << 25)) >> 26;
    let mut powers = [P, z, 0, 0, 0];
    for i in 2..5 {
        powers[i] = (powers[i - 1] * z + P / 2) >> 30;
    }
    let a = (A.iter().zip(powers).map(|(a, b)| a * b).sum::<i128>() + P / 2) >> 30;
    let b = (B.iter().zip(powers).map(|(a, b)| a * b).sum::<i128>() + P / 2) >> 30;
    let numerator = x * b + 2 * S * a;
    let denominator = 2 * b;
    let q = numerator.div_euclid(denominator);
    let r = numerator.rem_euclid(denominator);
    q + i128::from(2 * r > denominator || (2 * r == denominator && q & 1 != 0))
}
#[derive(Deserialize)]
struct Input {
    x: Vec<i64>,
    y: Option<Vec<i64>>,
}
#[derive(Clone, Serialize, Deserialize)]
struct Chunk {
    start: usize,
    count: usize,
    proof: String,
}
#[derive(Clone, Serialize, Deserialize)]
struct Public {
    version: String,
    count: usize,
    chunk_size: usize,
    commitments_x: Vec<String>,
    commitments_y: Vec<String>,
    chunks: Vec<Chunk>,
}
#[derive(Serialize, Deserialize)]
struct Private {
    version: String,
    output_y: Vec<i64>,
    blindings_x: Vec<String>,
    blindings_y: Vec<String>,
}
fn transcript(n: usize, start: usize, count: usize, chunk_size: usize) -> Transcript {
    let mut t = Transcript::new(b"siv-native-silu-proof");
    t.append_message(b"circuit", VERSION.as_bytes());
    for (label, value) in [
        (b"n".as_slice(), n),
        (b"start", start),
        (b"count", count),
        (b"chunk_size", chunk_size),
    ] {
        t.append_u64(label, value as u64);
    }
    t
}
fn point(encoded: &str) -> Result<CompressedRistretto> {
    let bytes: [u8; 32] = hex::decode(encoded)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("point length"))?;
    let point = CompressedRistretto(bytes);
    if point.decompress().is_none() {
        bail!("invalid Ristretto encoding");
    }
    Ok(point)
}
type GeneratorCache = Mutex<HashMap<usize, Arc<BulletproofGens>>>;
fn generators(cache: &GeneratorCache, multipliers: usize) -> Arc<BulletproofGens> {
    let capacity = multipliers.next_power_of_two();
    cache
        .lock()
        .unwrap()
        .entry(capacity)
        .or_insert_with(|| Arc::new(BulletproofGens::new(capacity, 1)))
        .clone()
}
fn pool() -> Result<rayon::ThreadPool> {
    let jobs = std::env::var("SIV_PROOF_JOBS")
        .ok()
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(4);
    if jobs == 0 {
        bail!("SIV_PROOF_JOBS must be positive");
    }
    Ok(rayon::ThreadPoolBuilder::new().num_threads(jobs).build()?)
}
fn prove(input: &Input, chunk_size: usize) -> Result<(Public, Private)> {
    if chunk_size == 0 || input.x.is_empty() {
        bail!("nonempty input and positive chunk size required");
    }
    let n = input.x.len();
    if input.x.iter().any(|&x| (x as i128).abs() > 16 * S + 1) {
        bail!("host input outside safe witness-building interval");
    }
    let y: Vec<i64> = input.y.clone().unwrap_or_else(|| {
        input
            .x
            .iter()
            .map(|&x| honest_y(x as i128) as i64)
            .collect()
    });
    if y.len() != n {
        bail!("x/y lengths differ");
    }
    if y.iter().any(|&y| (y as i128).abs() > 32 * S) {
        bail!("host output outside safe witness-building interval");
    }
    let pc = PedersenGens::default();
    let mut rng = OsRng;
    let rx: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
    let ry: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
    let mut public = Public {
        version: VERSION.into(),
        count: n,
        chunk_size,
        commitments_x: vec![],
        commitments_y: vec![],
        chunks: vec![],
    };
    for i in 0..n {
        public.commitments_x.push(hex::encode(
            pc.commit(scalar(input.x[i] as i128 + 16 * S), rx[i])
                .compress()
                .as_bytes(),
        ));
        public.commitments_y.push(hex::encode(
            pc.commit(scalar(y[i] as i128 + S), ry[i])
                .compress()
                .as_bytes(),
        ));
    }
    let cache = GeneratorCache::default();
    let starts: Vec<usize> = (0..n).step_by(chunk_size).collect();
    public.chunks = pool()?.install(|| {
        starts
            .par_iter()
            .map(|&start| -> Result<Chunk> {
                let count = chunk_size.min(n - start);
                let mut t = transcript(n, start, count, chunk_size);
                let mut prover = Prover::new(&pc, &mut t);
                let mut vars = Vec::with_capacity(count);
                for i in start..start + count {
                    let (_, x) = prover.commit(scalar(input.x[i] as i128 + 16 * S), rx[i]);
                    let (_, yvar) = prover.commit(scalar(y[i] as i128 + S), ry[i]);
                    vars.push((x, yvar));
                }
                for (j, (x, yvar)) in vars.into_iter().enumerate() {
                    let i = start + j;
                    constrain_silu(
                        &mut prover,
                        x,
                        yvar,
                        Some((input.x[i] as i128, y[i] as i128)),
                    )?;
                }
                let generators = generators(&cache, prover.metrics().multipliers);
                let proof = prover.prove(&generators)?;
                eprintln!("proved chunk start={start} count={count}");
                Ok(Chunk {
                    start,
                    count,
                    proof: hex::encode(proof.to_bytes()),
                })
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let private = Private {
        version: VERSION.into(),
        output_y: y,
        blindings_x: rx.iter().map(|x| hex::encode(x.to_bytes())).collect(),
        blindings_y: ry.iter().map(|x| hex::encode(x.to_bytes())).collect(),
    };
    Ok((public, private))
}
fn verify(public: &Public) -> Result<()> {
    if public.version != VERSION
        || public.count == 0
        || public.chunk_size == 0
        || public.commitments_x.len() != public.count
        || public.commitments_y.len() != public.count
    {
        bail!("invalid artifact metadata");
    }
    let pc = PedersenGens::default();
    let mut expected = 0;
    for chunk in &public.chunks {
        if expected >= public.count
            || chunk.count == 0
            || chunk.start != expected
            || chunk.count != public.chunk_size.min(public.count - expected)
        {
            bail!("invalid chunk schedule");
        }
        expected += chunk.count;
    }
    if expected != public.count {
        bail!("missing chunks");
    }
    let cache = GeneratorCache::default();
    pool()?.install(|| {
        public
            .chunks
            .par_iter()
            .try_for_each(|chunk| -> Result<()> {
                let mut t = transcript(public.count, chunk.start, chunk.count, public.chunk_size);
                let mut verifier = Verifier::new(&mut t);
                let mut vars = Vec::new();
                for i in chunk.start..chunk.start + chunk.count {
                    vars.push((
                        verifier.commit(point(&public.commitments_x[i])?),
                        verifier.commit(point(&public.commitments_y[i])?),
                    ));
                }
                for (x, y) in vars {
                    constrain_silu(&mut verifier, x, y, None)?;
                }
                let generators = generators(&cache, verifier.metrics().multipliers);
                verifier.verify(
                    &R1CSProof::from_bytes(&hex::decode(&chunk.proof)?)?,
                    &pc,
                    &generators,
                )?;
                Ok(())
            })
    })?;
    Ok(())
}
fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec(value)?)?;
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let started = Instant::now();
    match args.get(1).map(String::as_str) {
        Some("prove") => {
            let input: Input = serde_json::from_slice(&fs::read(
                args.get(2).context("prove INPUT_JSON OUTPUT_DIR [CHUNK]")?,
            )?)?;
            let out = Path::new(args.get(3).context("missing output directory")?);
            fs::create_dir_all(out)?;
            if out.join("private-blindings.json").exists() {
                bail!("output directory already contains private blindings");
            }
            let chunk = args.get(4).map(|v| v.parse()).transpose()?.unwrap_or(128);
            let (public, private) = prove(&input, chunk)?;
            write_json(out.join("public.json"), &public)?;
            let private_path = out.join("private-blindings.json");
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(&private_path)?;
                use std::io::Write;
                file.write_all(&serde_json::to_vec(&private)?)?;
            }
            #[cfg(not(unix))]
            write_json(private_path, &private)?;
            let proof_bytes: usize = public.chunks.iter().map(|c| c.proof.len() / 2).sum();
            println!(
                "{}",
                serde_json::json!({"values":public.count,"chunk_size":chunk,"multipliers_per_value":GATES_PER_VALUE,"commitment_bytes":64*public.count,"proof_bytes":proof_bytes,"seconds":started.elapsed().as_secs_f64()})
            );
        }
        Some("verify") => {
            let public: Public =
                serde_json::from_slice(&fs::read(args.get(2).context("verify PUBLIC_JSON")?)?)?;
            verify(&public)?;
            println!(
                "{}",
                serde_json::json!({"verified":true,"seconds":started.elapsed().as_secs_f64()})
            );
        }
        Some("metadata") => {
            let pc = PedersenGens::default();
            println!(
                "{}",
                serde_json::json!({"version":VERSION,"G":hex::encode(pc.B.compress().as_bytes()),"H":hex::encode(pc.B_blinding.compress().as_bytes()),"scalar_modulus":"7237005577332262213973186563042994240857116359379907606001950938285454250989","multipliers_per_value":GATES_PER_VALUE})
            );
        }
        Some("project") => {
            let public: Public = serde_json::from_slice(&fs::read(
                args.get(2)
                    .context("project PUBLIC_JSON PACKED_WEIGHTS OUTPUT_JSON")?,
            )?)?;
            if public.version != VERSION
                || public.count == 0
                || public.commitments_x.len() != public.count
                || public.commitments_y.len() != public.count
            {
                bail!("invalid projection metadata");
            }
            let weights = fs::read(args.get(3).context("missing weights file")?)?;
            let bit_count = 128usize
                .checked_mul(public.count)
                .context("projection size overflow")?;
            if weights.len() != bit_count.div_ceil(8) {
                bail!("expected128*n packed bits, LSB first, contiguous row-major");
            }
            let xs: Vec<RistrettoPoint> = public
                .commitments_x
                .iter()
                .map(|x| point(x).map(|p| p.decompress().unwrap()))
                .collect::<Result<_>>()?;
            let ys: Vec<RistrettoPoint> = public
                .commitments_y
                .iter()
                .map(|x| point(x).map(|p| p.decompress().unwrap()))
                .collect::<Result<_>>()?;
            let mut cx = Vec::new();
            let mut cy = Vec::new();
            for row in 0..128 {
                let mut x = RistrettoPoint::identity();
                let mut y = RistrettoPoint::identity();
                for col in 0..public.count {
                    let i = row * public.count + col;
                    if weights[i / 8] >> (i % 8) & 1 != 0 {
                        x += xs[col];
                        y += ys[col];
                    }
                }
                cx.push(hex::encode(x.compress().as_bytes()));
                cy.push(hex::encode(y.compress().as_bytes()));
            }
            write_json(
                args.get(4).context("missing projection output")?,
                &serde_json::json!({"commitments_x":cx,"commitments_y":cy}),
            )?;
            println!(
                "{}",
                serde_json::json!({"projected":true,"seconds":started.elapsed().as_secs_f64()})
            );
        }
        _ => {
            bail!("usage: siv-silu-proof prove INPUT_JSON OUTPUT_DIR [CHUNK] | verify PUBLIC_JSON")
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(x: Vec<i64>, y: Option<Vec<i64>>) -> Input {
        Input { x, y }
    }
    #[test]
    fn valid_endpoints_and_signs() {
        let p = prove(
            &fixture(
                vec![-16 * S as i64, -S as i64, 0, S as i64, 16 * S as i64],
                None,
            ),
            2,
        )
        .unwrap()
        .0;
        verify(&p).unwrap();
    }
    #[test]
    fn rejects_outside_domain() {
        let p = prove(&fixture(vec![16 * S as i64 + 1], None), 1).unwrap().0;
        assert!(verify(&p).is_err());
    }
    #[test]
    fn rejects_wrong_output() {
        let p = prove(&fixture(vec![0], Some(vec![10])), 1).unwrap().0;
        assert!(verify(&p).is_err());
    }
    #[test]
    fn rejects_malformed_metadata() {
        let p = prove(&fixture(vec![0], None), 1).unwrap().0;
        let mut bad = p.clone();
        bad.chunks.push(bad.chunks[0].clone());
        assert!(verify(&bad).is_err());
        let mut bad = p.clone();
        bad.chunks[0].count = usize::MAX;
        assert!(verify(&bad).is_err());
        let mut bad = p.clone();
        bad.chunks[0].count = 0;
        assert!(verify(&bad).is_err());
        let mut bad = p.clone();
        bad.chunks.clear();
        assert!(verify(&bad).is_err());
        let mut bad = p.clone();
        bad.chunk_size = 0;
        assert!(verify(&bad).is_err());
    }
    #[test]
    fn rejects_extreme_host_words() {
        for x in [i64::MIN, i64::MAX] {
            assert!(prove(&fixture(vec![x], None), 1).is_err());
        }
        assert!(prove(&fixture(vec![0], Some(vec![i64::MAX])), 1).is_err());
    }
    #[test]
    fn rejects_proof_and_commitment_tampering() {
        let p = prove(&fixture(vec![0], None), 1).unwrap().0;
        let mut bad = p.clone();
        let mut b = hex::decode(&bad.chunks[0].proof).unwrap();
        b[0] ^= 1;
        bad.chunks[0].proof = hex::encode(b);
        assert!(verify(&bad).is_err());
        let mut bad = p.clone();
        bad.commitments_x[0] = hex::encode(PedersenGens::default().B.compress().as_bytes());
        assert!(verify(&bad).is_err());
    }
}
