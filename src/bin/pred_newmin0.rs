#![allow(unused_mut)]
#![allow(unused_imports)]
#![allow(non_snake_case)]
#![allow(unused_imports)]

//extern crate csv;
//extern crate ndarray;
//extern crate ndarray_csv;

use concrete::*;

use csv::{ReaderBuilder};
use ndarray::{Array, Array2};
use ndarray_csv::{Array2Reader};
use std::error::Error;
use std::fs::File;

//use glob::glob;

fn sum_vector_with_static_encoder(x: &VectorLWE) -> VectorLWE{
    let min = x.encoders[0].get_min() as f64;
    let delta = x.encoders[0].delta as f64;
    let x_sum = x.sum_with_new_min(min).unwrap();
    let x_mod = x_sum.add_constant_static_encoder(&vec![0.9999f64 * delta]).unwrap();
    return x_mod;
}  

fn sigmoid(v: f64) -> f64 {
    if v < -40.0 {
        0.0
    } else if v > 40.0 {
        1.0
    } else {
        1.0 / (1.0 + f64::exp(-1.*v))
    }
}    

fn main() -> Result<(), Box<dyn Error>> {
    let path = "keys";
    println!("Load parameters for linear regression ... ");
    let file = File::open("data/coeff.csv")?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let coeff: Array2<f64> = reader.deserialize_array2_dynamic().unwrap();
    let coeff = coeff.column(0).to_vec();
    println!("Number of coefficients {}",coeff.len());
    
    let file = File::open("data/intercept.csv")?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let intercept: Array2<f64> = reader.deserialize_array2_dynamic().unwrap();
    let intercept = intercept.column(0).to_vec();
    println!("Intercept {:?}",intercept.len());
        
    let max_constant: f64 = 4.0;
    let nb_bit_padding = 8;
    let N = 197; 

    // To execute sigmoid we need to set-up for bootstrapping
    //let enc = Encoder::new(0.0, 1.0, 4, 1).unwrap();
    //println!("Load Bootstrapping Key 01 ... \n");
    //let bsk01_path = format!("{}/bsk01_LWE.json", path);
    //let bsk01 = LWEBSK::load(&bsk01_path);    
        
    // This is only for debugging
    // **************************
    println!("DEBUG: Load LWE secret key ... \n");
    let sk0_LWE_path = format!("{}/sk0_LWE.json",path);
    let sk0 = LWESecretKey::load(&sk0_LWE_path).unwrap();    
    //let sk1_LWE_path = format!("{}/sk1_LWE.json",path);
    //let sk1 = LWESecretKey::load(&sk1_LWE_path).unwrap();       
    
    println!("DEBUG: Load ground truth data ... \n");
    let file = File::open("data/y_test.csv").expect("could not read data file");
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let y_test: Array2<f64> = reader.deserialize_array2_dynamic().expect("conversion error");
    //let N = y_test.shape()[0];
    // **************************
    
    println!("Number of data rows: {}", N);
    
    for i in 0..N {        
        let encfile = format!("data/X_test0/{}.enc",i);
        println!("{}", encfile);
        
        let features = VectorLWE::load(&encfile).unwrap();
        let terms = features.mul_constant_with_padding(&coeff, max_constant, nb_bit_padding)?;
        let temps = sum_vector_with_static_encoder(&terms);
        let z_pred = temps.add_constant_dynamic_encoder(&intercept)?;
        
        // we stop here and pass the linear prediction z
        // instead of calculating the sigmoid
        // and let the prediction remain on the real line
        
        let y_pred = z_pred.clone();
        
        // This is only for debugging
        // **************************
        if i<5 {
            //features.pp();
            //println!("DEBUG: terms {:?}", terms.decrypt_decode(&sk0).unwrap());
            //terms.pp();
            println!("DEBUG: temps {:?}", temps.decrypt_decode(&sk0).unwrap()); 
            //temps.pp();
            println!("DEBUG: z_pred {:?}", z_pred.decrypt_decode(&sk0).unwrap()); 
            //price.pp();
            println!("DEBUG: s(z) {:?}", sigmoid(z_pred.decrypt_decode(&sk0).unwrap()[0])); 
            //price.pp();
            println!("DEBUG: y_pred {:?}", y_pred.decrypt_decode(&sk0).unwrap()); 
            //price.pp();
            println!("DEBUG: y_test {:?}", y_test.row(i).to_vec());
        } else if false {
            break;
        };        
        // **************************

        let encfile = format!("data/y_test0/{}.enc",i);
        y_pred.save(&encfile).unwrap();
    };
    
    Ok(())
}