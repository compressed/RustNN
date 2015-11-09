#![feature(test)]

extern crate nn;
extern crate time;
extern crate test;
extern crate rand;

use nn::{NN, HaltCondition, LearningMode};
use test::Bencher;
use rand::distributions::{IndependentSample, Range};

const INPUT_SIZE: u32 = 10_000;

#[bench]
fn single_threaded(b: &mut Bencher) {
    let examples = get_examples();

    b.iter(|| {
        let mut net = NN::new(&[INPUT_SIZE,2,50,10]);

        net.train(&examples)
            .halt_condition( HaltCondition::Epochs(1) )
            .learning_mode( LearningMode::Incremental )
            .momentum(0.1)
            .go();
    })
}

#[bench]
fn multi_threaded(b: &mut Bencher) {
    let examples = get_examples();

    b.iter(|| {
        let mut net = NN::new(&[INPUT_SIZE,2,50,10]);

        net.train(&examples)
            .halt_condition( HaltCondition::Epochs(1) )
            .learning_mode( LearningMode::Chunk )
            .momentum(0.1)
            .num_threads(3)
            .chunk_size(100)
            .go();
    })
}

fn get_examples() -> Vec<(Vec<f64>, Vec<f64>)> {
    let between = Range::new(-1.0, 1.0);
    let mut rng = rand::thread_rng();

    let mut examples = Vec::with_capacity(100);

    for _ in 0..100 {
        let input = (0..INPUT_SIZE).map(|_| between.ind_sample(&mut rng)).collect::<Vec<f64>>();
        let output = (0..10).map(|_| between.ind_sample(&mut rng)).collect::<Vec<f64>>();
        examples.push((input, output));
    }

    examples
}
