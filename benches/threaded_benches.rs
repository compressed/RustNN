#![feature(test)]

extern crate nn;
extern crate time;
extern crate test;

use nn::{NN, HaltCondition, LearningMode};
use test::Bencher;

#[bench]
fn single_threaded(b: &mut Bencher) {
    let examples = get_examples();

    b.iter(|| {
        // create a new neural network
        let mut net = NN::new(&[2,3,3,1]);

        net.train(&examples)
            .halt_condition( HaltCondition::Epochs(5) )
            .learning_mode( LearningMode::Incremental )
            .momentum(0.1)
            .go();
    })
}

#[bench]
fn multi_threaded(b: &mut Bencher) {
    let examples = get_examples();

    b.iter(|| {
        // create a new neural network
        let mut net = NN::new(&[2,3,3,1]);

        net.train(&examples)
            .halt_condition( HaltCondition::Epochs(5) )
            .learning_mode( LearningMode::Incremental )
            .momentum(0.1)
            .num_threads(4)
            .go();
    })
}

fn get_examples() -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut examples = vec![];

    for _ in 0..100 {
        examples.push((vec![0f64, 0f64], vec![0f64]));
        examples.push((vec![0f64, 1f64], vec![1f64]));
        examples.push((vec![1f64, 0f64], vec![1f64]));
        examples.push((vec![1f64, 1f64], vec![0f64]));
    }

    examples
}
