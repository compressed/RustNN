//! An easy to use neural network library written in Rust.
//!
//! # Description
//! nn is a [feedforward neural network ](http://en.wikipedia.org/wiki/Feedforward_neural_network)
//! library. The library
//! generates fully connected multi-layer artificial neural networks that
//! are trained via [backpropagation](http://en.wikipedia.org/wiki/Backpropagation).
//! Networks are trained using an incremental training mode.
//!
//! # XOR example
//!
//! This example creates a neural network with `2` nodes in the input layer,
//! a single hidden layer containing `3` nodes, and `1` node in the output layer.
//! The network is then trained on examples of the `XOR` function. All of the
//! methods called after `train(&examples)` are optional and are just used
//! to specify various options that dictate how the network should be trained.
//! When the `go()` method is called the network will begin training on the
//! given examples. See the documentation for the `NN` and `Trainer` structs
//! for more details.
//!
//! ```rust
//! use nn::{NN, HaltCondition};
//!
//! // create examples of the XOR function
//! // the network is trained on tuples of vectors where the first vector
//! // is the inputs and the second vector is the expected outputs
//! let examples = [
//!     (vec![0f64, 0f64], vec![0f64]),
//!     (vec![0f64, 1f64], vec![1f64]),
//!     (vec![1f64, 0f64], vec![1f64]),
//!     (vec![1f64, 1f64], vec![0f64]),
//! ];
//!
//! // create a new neural network by passing a pointer to an array
//! // that specifies the number of layers and the number of nodes in each layer
//! // in this case we have an input layer with 2 nodes, one hidden layer
//! // with 3 nodes and the output layer has 1 node
//! let mut net = NN::new(&[2, 3, 1]);
//!
//! // train the network on the examples of the XOR function
//! // all methods seen here are optional except go() which must be called to begin training
//! // see the documentation for the Trainer struct for more info on what each method does
//! net.train(&examples)
//!     .halt_condition( HaltCondition::Epochs(10000) )
//!     .log_interval( Some(100) )
//!     .momentum( 0.1 )
//!     .rate( 0.3 )
//!     .go();
//!
//! // evaluate the network to see if it learned the XOR function
//! for &(ref inputs, ref outputs) in examples.iter() {
//!     let results = net.run(inputs);
//!     let (result, key) = (results[0].round(), outputs[0]);
//!     assert!(result == key);
//! }
//! ```

extern crate rand;
extern crate rustc_serialize;
extern crate time;
extern crate crossbeam;
extern crate simple_parallel;
extern crate env_logger;
#[macro_use] extern crate log;

use simple_parallel::Pool;
use HaltCondition::{ Epochs, MSE, Timer };
use LearningMode::{ Incremental, Chunk };
use std::iter::{Zip, Enumerate};
use std::slice;
use rustc_serialize::json;
use time::{ Duration, PreciseTime };
use rand::Rng;
use std::sync::RwLock;

const DEFAULT_LEARNING_RATE: f64 = 0.3f64;
const DEFAULT_MOMENTUM: f64 = 0f64;
const DEFAULT_EPOCHS: u32 = 1000;
/// Default chunk size for splitting examples up during multi-threaded training
/// the larger this number, the more examples are tests before actually updating the weights
/// of the model
const DEFAULT_CHUNK_SIZE: usize = 10;

/// Specifies when to stop training the network
#[derive(Debug, Copy, Clone)]
pub enum HaltCondition {
    /// Stop training after a certain number of epochs
    Epochs(u32),
    /// Train until a certain error rate is achieved
    MSE(f64),
    /// Train for some fixed amount of time and then halt
    Timer(Duration),
}

/// Specifies which [learning mode](http://en.wikipedia.org/wiki/Backpropagation#Modes_of_learning)
/// to use when training the network
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LearningMode {
    /// train the network Incrementally (updates weights after each example)
    Incremental,
    /// train the network in chunks, weights are updated after each chunk has been procoessed
    ///
    /// This is useful during multi-threaded training
    Chunk,
}

/// Used to specify options that dictate how a network will be trained
#[derive(Debug)]
pub struct Trainer<'a,'b> {
    examples: &'b [(Vec<f64>, Vec<f64>)],
    rate: f64,
    momentum: f64,
    log_interval: Option<u32>,
    halt_condition: HaltCondition,
    learning_mode: LearningMode,
    nn: &'a mut NN,
    num_threads: usize,
    /// number of iterations to process in parallel before updating weights
    /// useful in multi-threading
    chunk_size: usize,
}

/// `Trainer` is used to chain together options that specify how to train a network.
/// All of the options are optional because the `Trainer` struct
/// has default values built in for each option. The `go()` method must
/// be called however or the network will not be trained.
impl<'a,'b> Trainer<'a,'b>  {

    /// Specifies the learning rate to be used when training (default is `0.3`)
    /// This is the step size that is used in the backpropagation algorithm.
    pub fn rate(&mut self, rate: f64) -> &mut Trainer<'a,'b> {
        assert!(rate > 0.0, "the learning rate must be a positive number");

        self.rate = rate;
        self
    }

    /// Specifies the momentum to be used when training (default is `0.0`)
    pub fn momentum(&mut self, momentum: f64) -> &mut Trainer<'a,'b> {
        assert!(momentum > 0.0, "momentum must be a positive number");

        self.momentum = momentum;
        self
    }

    /// Specifies how often (measured in batches) to log the current error rate (mean squared error) during training.
    /// `Some(x)` means log after every `x` batches and `None` means never log
    pub fn log_interval(&mut self, log_interval: Option<u32>) -> &mut Trainer<'a,'b> {
        match log_interval {
            Some(interval) if interval < 1 => {
                panic!("log interval must be Some positive number or None")
            }
            _ => ()
        }

        self.log_interval = log_interval;
        self
    }

    /// Specifies when to stop training. `Epochs(x)` will stop the training after
    /// `x` epochs (one epoch is one loop through all of the training examples)
    /// while `MSE(e)` will stop the training when the error rate
    /// is at or below `e`. `Timer(d)` will halt after the [duration](https://doc.rust-lang.org/time/time/struct.Duration.html) `d` has
    /// elapsed.
    pub fn halt_condition(&mut self, halt_condition: HaltCondition) -> &mut Trainer<'a,'b> {
        match halt_condition {
            Epochs(epochs) => assert!(epochs > 0, "must train for at least one epoch"),
            MSE(mse) => assert!(mse > 0.0, "MSE must be greater than 0"),
            _ => {}
        }

        self.halt_condition = halt_condition;
        self
    }
    /// Specifies what [mode](http://en.wikipedia.org/wiki/Backpropagation#Modes_of_learning) to train the network in.
    /// `Incremental` means update the weights in the network after every example.
    pub fn learning_mode(&mut self, learning_mode: LearningMode) -> &mut Trainer<'a,'b> {
        self.learning_mode = learning_mode;
        self
    }

    /// Specifies the number of threads to use for training
    pub fn num_threads(&mut self, num_threads: usize) -> &mut Trainer<'a,'b> {
        self.num_threads = num_threads;
        self
    }

    /// Specifies the number of threads to use for training
    pub fn chunk_size(&mut self, chunk_size: usize) -> &mut Trainer<'a,'b> {
        self.chunk_size = chunk_size;
        self
    }

    /// When `go` is called, the network will begin training based on the
    /// options specified. If `go` does not get called, the network will not
    /// get trained!
    pub fn go(&mut self) -> f64 {
        self.nn.train_details(
            self.examples,
            self.rate,
            self.momentum,
            self.log_interval,
            self.halt_condition,
            self.num_threads,
            self.learning_mode,
            self.chunk_size,
        )
    }

}

/// Neural network
#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct NN {
    layers: Vec<Vec<Vec<f64>>>,
    num_inputs: u32,
}

impl NN {
    /// Each number in the `layers_sizes` parameter specifies a
    /// layer in the network. The number itself is the number of nodes in that
    /// layer. The first number is the input layer, the last
    /// number is the output layer, and all numbers between the first and
    /// last are hidden layers. There must be at least two layers in the network.
    pub fn new(layers_sizes: &[u32]) -> NN {
        let mut rng = rand::thread_rng();

        assert!(layers_sizes.len() >= 2, "must have atleast two layers");

        assert!(layers_sizes.iter().find(|x| &&0 == x).is_none(), "can't have any empty layers");

        let mut layers = Vec::new();
        let mut it = layers_sizes.iter();
        // get the first layer size
        let first_layer_size = *it.next().unwrap();

        // setup the rest of the layers
        let mut prev_layer_size = first_layer_size;
        for &layer_size in it {
            let mut layer: Vec<Vec<f64>> = Vec::new();
            for _ in 0..layer_size {
                let mut node: Vec<f64> = Vec::new();
                for _ in 0..prev_layer_size+1 {
                    let random_weight: f64 = rng.gen_range(-0.5f64, 0.5f64);
                    node.push(random_weight);
                }
                node.shrink_to_fit();
                layer.push(node)
            }
            layer.shrink_to_fit();
            layers.push(layer);
            prev_layer_size = layer_size;
        }
        layers.shrink_to_fit();
        NN { layers: layers, num_inputs: first_layer_size }
    }

    /// Runs the network on an input and returns a vector of the results.
    /// The number of `f64`s in the input must be the same
    /// as the number of input nodes in the network. The length of the results
    /// vector will be the number of nodes in the output layer of the network.
    pub fn run(&self, inputs: &[f64]) -> Vec<f64> {
        if inputs.len() as u32 != self.num_inputs {
            panic!("input has a different length than the network's input layer");
        }
        self.do_run(inputs).pop().unwrap()
    }

    /// Takes in vector of examples and returns a `Trainer` struct that is used
    /// to specify options that dictate how the training should proceed.
    /// No actual training will occur until the `go()` method on the
    /// `Trainer` struct is called.
    pub fn train<'b>(&'b mut self, examples: &'b [(Vec<f64>, Vec<f64>)]) -> Trainer {
        Trainer {
            examples: examples,
            rate: DEFAULT_LEARNING_RATE,
            momentum: DEFAULT_MOMENTUM,
            log_interval: None,
            halt_condition: Epochs(DEFAULT_EPOCHS),
            learning_mode: Incremental,
            nn: self,
            num_threads: 1,
            chunk_size: DEFAULT_CHUNK_SIZE,
        }
    }

    /// Encodes the network as a JSON string.
    pub fn to_json(&self) -> String {
        json::encode(self).ok().expect("encoding JSON failed")
    }

    /// Builds a new network from a JSON string.
    pub fn from_json(encoded: &str) -> NN {
        let network: NN = json::decode(encoded).ok().expect("decoding JSON failed");
        network
    }

    fn train_details(&mut self, examples: &[(Vec<f64>, Vec<f64>)], rate: f64, momentum: f64,
                     log_interval: Option<u32>, halt_condition: HaltCondition, num_threads: usize,
                     learning_mode: LearningMode, chunk_size: usize) -> f64 {
        // check that input and output sizes are correct
        let input_layer_size = self.num_inputs;
        let output_layer_size = self.layers[self.layers.len() - 1].len();
        for &(ref inputs, ref outputs) in examples.iter() {
            if inputs.len() as u32 != input_layer_size {
                panic!("input has a different length than the network's input layer");
            }
            if outputs.len() != output_layer_size {
                panic!("output has a different length than the network's output layer");
            }
        }

        match learning_mode {
            Incremental => {
                assert!(num_threads == 1, "incremental training can only be single-threaded");
                self.train_incremental(examples, rate, momentum, log_interval, halt_condition)
            },
            Chunk => self.train_chunk(examples, rate, momentum, log_interval, halt_condition, num_threads, chunk_size),
        }
    }

    fn train_incremental(&mut self, examples: &[(Vec<f64>, Vec<f64>)], rate: f64, momentum: f64, log_interval: Option<u32>,
                    halt_condition: HaltCondition) -> f64 {
        let mut prev_deltas = self.make_weights_tracker(0.0f64);
        let mut epochs = 0u32;
        let start_time = PreciseTime::now();
        let mut training_error_rate = 0f64;
        loop {
            if epochs > 0 {
                // log error rate if necessary
                match log_interval {
                    Some(interval) if epochs % interval == 0 => {
                        println!("error rate: {}", training_error_rate);
                    },
                    _ => (),
                }

                // check if we've met the halt condition yet
                match halt_condition {
                    Epochs(epochs_halt) => {
                        if epochs == epochs_halt { break }
                    },
                    MSE(target_error) => {
                        if training_error_rate <= target_error { break }
                    },
                    Timer(duration) => {
                        let now = PreciseTime::now();
                        if start_time.to(now) >= duration { break }
                    }
                }
            }

            training_error_rate = 0f64;
            for &(ref inputs, ref targets) in examples.iter() {
                let results = self.do_run(&inputs);
                let weight_updates = self.calculate_weight_updates(&results, &targets);
                training_error_rate += calculate_error(&results, &targets);
                self.update_weights(&weight_updates, &mut prev_deltas, rate, momentum)
            }
            epochs += 1;
        }
        training_error_rate
    }

    fn train_chunk(&mut self, examples: &[(Vec<f64>, Vec<f64>)], rate: f64, momentum: f64, log_interval: Option<u32>,
                    halt_condition: HaltCondition, num_threads: usize, chunk_size: usize) -> f64 {
        let mut prev_deltas = self.make_weights_tracker(0.0f64);
        let mut epochs = 0u32;
        let start_time = PreciseTime::now();
        let mut training_error_rate = 0f64;
        let mut pool = Pool::new(num_threads);
        let self_lock = RwLock::new(self);

        loop {
            if epochs > 0 {
                // log error rate if necessary
                match log_interval {
                    Some(interval) if epochs % interval == 0 => {
                        println!("error rate: {}", training_error_rate);
                    },
                    _ => (),
                }

                // check if we've met the halt condition yet
                match halt_condition {
                    Epochs(epochs_halt) => {
                        if epochs == epochs_halt { break }
                    },
                    MSE(target_error) => {
                        if training_error_rate <= target_error { break }
                    },
                    Timer(duration) => {
                        let now = PreciseTime::now();
                        if start_time.to(now) >= duration { break }
                    }
                }
            }

            training_error_rate = 0f64;

            let mut error_weights = (0.0, vec![vec![vec![]]]);
            crossbeam::scope(|scope| {
                for exs in examples.chunks(chunk_size) {
                    error_weights = pool.unordered_map(scope, exs, |&(ref inputs, ref targets)| {
                        let results = self_lock.read().unwrap().do_run(&inputs);
                        let weight_updates = self_lock.read().unwrap().calculate_weight_updates(&results, &targets);
                        (calculate_error(&results, &targets), weight_updates)
                    }).fold((0.0, self_lock.read().unwrap().make_weights_tracker(0.0)), |(mut orig_err, mut orig_weights), (_, (new_err, new_weights))| {
                        orig_err += new_err;
                        sum_weights(&mut orig_weights, new_weights);
                        (orig_err, orig_weights)
                    });

                    let (err, ref weight_updates) = error_weights;
                    info!("err={:?}", err);
                    training_error_rate += err;
                    self_lock.write().unwrap().update_weights(&weight_updates, &mut prev_deltas, rate, momentum);
                }
            });
            epochs += 1;
        }
        training_error_rate
    }

    fn do_run(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut results = Vec::new();
        results.push(inputs.to_vec());
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let mut layer_results = Vec::new();
            for node in layer.iter() {
                layer_results.push( sigmoid(modified_dotprod(&node, &results[layer_index])) )
            }
            results.push(layer_results);
        }
        results
    }

    // updates all weights in the network
    fn update_weights(&mut self, network_weight_updates: &Vec<Vec<Vec<f64>>>, prev_deltas: &mut Vec<Vec<Vec<f64>>>, rate: f64, momentum: f64) {
        for layer_index in 0..self.layers.len() {
            let mut layer = &mut self.layers[layer_index];
            let layer_weight_updates = &network_weight_updates[layer_index];
            for node_index in 0..layer.len() {
                let mut node = &mut layer[node_index];
                let node_weight_updates = &layer_weight_updates[node_index];
                for weight_index in 0..node.len() {
                    let weight_update = node_weight_updates[weight_index];
                    let prev_delta = prev_deltas[layer_index][node_index][weight_index];
                    let delta = (rate * weight_update) + (momentum * prev_delta);
                    node[weight_index] += delta;
                    prev_deltas[layer_index][node_index][weight_index] = delta;
                }
            }
        }

    }

    // calculates all weight updates by backpropagation
    fn calculate_weight_updates(&self, results: &Vec<Vec<f64>>, targets: &[f64]) -> Vec<Vec<Vec<f64>>> {
        let layers = &self.layers;
        let mut network_errors:Vec<Vec<f64>> = Vec::with_capacity(layers.len());
        let mut network_weight_updates = Vec::with_capacity(layers.len());
        let network_results = &results[1..]; // skip the input layer
        let mut next_layer_nodes: Option<&Vec<Vec<f64>>> = None;

        for (layer_index, (layer_nodes, layer_results)) in iter_zip_enum(layers, network_results).rev() {
            let prev_layer_results = &results[layer_index];
            let mut layer_errors = Vec::with_capacity(layer_results.len());
            let mut layer_weight_updates = Vec::with_capacity(layer_results.len());


            for (node_index, (node, &result)) in iter_zip_enum(layer_nodes, layer_results) {
                let mut node_weight_updates = Vec::with_capacity(node.len());

                // calculate error for this node
                let node_error = match layer_index {
                    s if s == layers.len() -1 => result * (1f64 - result) * (targets[node_index] - result),
                    _ => {
                        let mut sum = 0f64;
                        let next_layer_errors = &network_errors[network_errors.len() - 1];
                        for (next_node, &next_node_error_data) in next_layer_nodes.unwrap().iter().zip((next_layer_errors).iter()) {
                            sum += next_node[node_index+1] * next_node_error_data; // +1 because the 0th weight is the threshold
                        }
                        result * (1f64 - result) * sum
                    }
                };

                // calculate weight updates for this node
                for weight_index in 0..node.len() {
                    let prev_layer_result = match weight_index {
                        0 => 1f64, // threshold
                        _ => prev_layer_results[weight_index-1],
                    };

                    let weight_update = node_error * prev_layer_result;
                    node_weight_updates.push(weight_update);
                }

                layer_errors.push(node_error);
                layer_weight_updates.push(node_weight_updates);
            }

            network_errors.push(layer_errors);
            network_weight_updates.push(layer_weight_updates);
            next_layer_nodes = Some(&layer_nodes);
        }

        // updates were built by backpropagation so reverse them
        network_weight_updates.reverse();
        network_weight_updates
    }

    fn make_weights_tracker<T: Clone>(&self, place_holder: T) -> Vec<Vec<Vec<T>>> {
        let mut network_level = Vec::new();
        for layer in self.layers.iter() {
            let mut layer_level = Vec::new();
            for node in layer.iter() {
                let mut node_level = Vec::new();
                for _ in node.iter() {
                    node_level.push(place_holder.clone());
                }
                layer_level.push(node_level);
            }
            network_level.push(layer_level);
        }

        network_level
    }
}

fn modified_dotprod(node: &Vec<f64>, values: &Vec<f64>) -> f64 {
    let mut it = node.iter();
    let mut total = *it.next().unwrap(); // start with the threshold weight
    for (weight, value) in it.zip(values.iter()) {
        total += weight * value;
    }
    total
}

fn sigmoid(y: f64) -> f64 {
    1f64 / (1f64 + (-y).exp())
}


// takes two arrays and enumerates the iterator produced by zipping each of
// their iterators together
fn iter_zip_enum<'s, 't, S: 's, T: 't>(s: &'s [S], t: &'t [T]) ->
    Enumerate<Zip<slice::Iter<'s, S>, slice::Iter<'t, T>>>  {
    s.iter().zip(t.iter()).enumerate()
}

// calculates MSE of output layer
fn calculate_error(results: &Vec<Vec<f64>>, targets: &[f64]) -> f64 {
    let ref last_results = results[results.len()-1];
    let mut total:f64 = 0f64;
    for (&result, &target) in last_results.iter().zip(targets.iter()) {
        total += (target - result).powi(2);
    }
    total / (last_results.len() as f64)
}

fn sum_weights(orig_weights: &mut Vec<Vec<Vec<f64>>>, new_weights: Vec<Vec<Vec<f64>>>) {
    orig_weights.iter_mut().enumerate().map(|(i1, v1)| {
        v1.iter_mut().enumerate().map(|(i2, v2)| {
            v2.iter_mut().enumerate().map(|(i3, v3)| {
                *v3 += new_weights[i1][i2][i3];
            }).last();
        }).last();
    }).last();
}
