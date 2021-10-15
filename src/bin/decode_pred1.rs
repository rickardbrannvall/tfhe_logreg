#![allow(unused_mut)]
#![allow(unused_imports)]
#![allow(non_snake_case)]
#![allow(unused_imports)]

//extern crate csv;
//extern crate ndarray;
//extern crate ndarray_csv;

use concrete::*;

use csv::{ReaderBuilder, WriterBuilder};
use ndarray::{Array, Array2};
use ndarray_csv::{Array2Reader, Array2Writer};
use std::error::Error;
use std::fs::File;

// replace vector format with scalar

fn main() -> Result<(), CryptoAPIError> {

    println!("Load LWE secret key ... ");
    let path = "keys";
    let sk1_LWE_path = format!("{}/sk1_LWE.json",path);
    let sk1 = LWESecretKey::load(&sk1_LWE_path).unwrap();    

    println!("Load ground truth data ... ");
    let file = File::open("data/y_test.csv").expect("could not read data file");
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let y_test: Array2<f64> = reader.deserialize_array2_dynamic().expect("conversion error");
    //println!("{:?}",y_test.dim());

    let N = y_test.shape()[0];
    println!("Number of data rows: {}", N);
    
    let mut y_pred = y_test.clone();
    
    println!("Load and decrypt predictions ...");
    for i in 0..N {        
        let encfile = format!("data/y_test1/{}.enc",i);
        let pred = VectorLWE::load(&encfile).unwrap();
        
        // for prediction it is sufficient to check that z = a x + b >= 0, i.e. no non-linearity
        //y_pred[[i,0]] = if pred.decrypt_decode(&sk0).unwrap()[0] >= 0.0 {1.0} else {0.0};
        
        // but one can also make the check on y = sigmoid(a x + b) >= 0.5 instead 
        y_pred[[i,0]] = if pred.decrypt_decode(&sk1).unwrap()[0] >= 0.5 {1.0} else {0.0};

        if i<20 {
            println!("{}",y_pred[[i,0]]); 
        } else if false {
            break;
        };
    };
 
    //println!("{:?}",y_pred);
    
    let acc = (&y_pred-&y_test).mapv(|x| {
        if x*x < 0.00001 {
            1.0
        } else {
            0.0
        }
    }).sum()/y_pred.len() as f64;

    println!("Accuracy: {}", acc);

    let file = File::create("data/y_test0/y_test.csv").expect("could not create file");
    let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
    writer.serialize_array2(&y_pred).expect("could not write file");    
        
    Ok(())
}
