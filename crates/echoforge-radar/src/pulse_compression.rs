use crate::ComplexSample;

pub fn matched_filter(received: &[ComplexSample], reference: &[ComplexSample]) -> Vec<ComplexSample> {
    if received.is_empty() || reference.is_empty() {
        return Vec::new();
    }

    let mut output = vec![ComplexSample::new(0.0, 0.0); received.len() + reference.len() - 1];
    let reference_conj_rev: Vec<_> = reference.iter().rev().map(|sample| sample.conj()).collect();

    for (i, sample) in received.iter().enumerate() {
        for (j, ref_sample) in reference_conj_rev.iter().enumerate() {
            output[i + j] += *sample * *ref_sample;
        }
    }

    output
}

pub fn pulse_compress(received: &[ComplexSample], reference: &[ComplexSample]) -> Vec<ComplexSample> {
    matched_filter(received, reference)
}

pub fn magnitude(samples: &[ComplexSample]) -> Vec<f32> {
    samples.iter().map(|sample| sample.norm()).collect()
}

