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
    let N = 197; //102;

    // This is only for debugging
    println!("DEBUG: Load LWE secret key ... \n");
    let path = "keys";
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
        let y_pred = temps.add_constant_dynamic_encoder(&intercept)?;
        
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

        let encfile = format!("data/y_test0/{}.enc",i);
        y_pred.save(&encfile).unwrap();
    };
    
    Ok(())
}