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

fn main() -> Result<(), Box<dyn Error>> {
    //let path = "keys_80_2048_60_64";
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
    
    //let offset = vec![intercept[0]/coeff.len() as f64; coeff.len()];    
    
    let max_constant: f64 = 4.0;
    let nb_bit_padding = 8;
    let N = 3; //197;
    
    
    // To execute sigmoid we need to set-up for bootstrapping
    
    let enc = Encoder::new(0., 1., 8, 1)?;

    println!("Load Bootstrapping Key 01 ... \n");
    let bsk01_path = format!("{}/bsk01_LWE.json", path);
    let bsk01 = LWEBSK::load(&bsk01_path);    

    fn sigmoid(v: f64) -> f64 {
      if v < -40.0 {
          0.0
      } else if v > 40.0 {
          1.0
      } else {
          1.0 / (1.0 + f64::exp(-v))
      }
    }    

    
    // This is only for debugging
    println!("DEBUG: Load LWE secret key ... \n");
    let sk0_LWE_path = format!("{}/sk0_LWE.json",path);
    let sk0 = LWESecretKey::load(&sk0_LWE_path).unwrap();    
    
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
        //let temps = terms.add_constant_static_encoder(&offset)?; 
        //let price = temps.sum_with_new_min(0.).unwrap();
        let temps = terms.sum_with_new_min(-30.).unwrap(); 
        let z_pred = temps.add_constant_dynamic_encoder(&intercept)?;
        
        let y_pred = z_pred.bootstrap_nth_with_function(&bsk01, sigmoid, &enc, 0)?;
        
        // This is only for debugging
        if true && i<5 {
            //features.pp();
            println!("DEBUG: terms {:?}", terms.decrypt_decode(&sk0).unwrap());
            //terms.pp();
            println!("DEBUG: temps {:?}", temps.decrypt_decode(&sk0).unwrap()); 
            //temps.pp();
            println!("DEBUG: y_pred {:?}", y_pred.decrypt_decode(&sk0).unwrap()); 
            //price.pp();
            println!("DEBUG: y_test {:?}", y_test.row(i).to_vec());
        }
        // **************************

        let encfile = format!("data/y_test1/{}.enc",i);
        y_pred.save(&encfile).unwrap();
    };
    
    Ok(())
}