#![allow(non_snake_case)]
#![allow(unused_imports)]

use concrete::*;
use std::path::Path;
//use std::fs::File;
//use std::io::Read;
use std::str::FromStr;
use std::fs;

use csv::{ReaderBuilder, WriterBuilder};
use ndarray::{Array, Array2};
use ndarray_csv::{Array2Reader, Array2Writer};
use std::error::Error;
use std::fs::File;


// replace vector format with scalar

fn main() -> Result<(), CryptoAPIError> {
    let path = "keys";
    
    println!("loading LWE key... \n");
    let sk0_LWE_path = format!("{}/sk0_LWE.json",path);
    let sk0 = LWESecretKey::load(&sk0_LWE_path).unwrap();    
    
    // create an encoder
    let enc = Encoder::new(-7.5, 7.5, 8, 8)?;
    
    // Process test set
    
    let file = File::open("data/X_test.csv").expect("could not read data file");
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let X_test: Array2<f64> = reader.deserialize_array2_dynamic().expect("conversion error");
    println!("{:?}",X_test.dim());

    let N = X_test.shape()[0];
    //let N = 5;
    println!("Number of data rows: {}", N);
    
    for i in 0..N {
        let x = X_test.row(i).to_vec();
        let encfile = format!("data/X_test0/{}.enc",i);
        println!("write {}",encfile);
        let c = VectorLWE::encode_encrypt(&sk0, &x, &enc).unwrap();
        c.save(&encfile).unwrap();
    }
    
    // Process train set
    
    /*
    let file = File::open("data/X_test.csv").expect("could not read data file");
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let X_test: Array2<f64> = reader.deserialize_array2_dynamic().expect("conversion error");
    println!("{:?}",X_test.dim());
    
    let file = File::open("data/y_test.csv").expect("could not read data file");
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let y_test: Array2<f64> = reader.deserialize_array2_dynamic().expect("conversion error");
    println!("{:?}",y_test.dim());

    let N = y_test.shape()[0];
    //let N = X_test.shape()[0];
    //let N = 5;
    println!("Number of data rows: {}", N);
    
    for i in 0..N {
        let x = X_test.row(i).to_vec();
        //println!("write {:?}", x);
        let encfile = format!("data/X_test/{}.enc",i);
        println!("write {}",encfile);
        let c = VectorLWE::encode_encrypt(&sk0, &x, &enc).unwrap();
        c.save(&encfile).unwrap();
        let y = y_test.row(i).to_vec();
        //println!("write {:?}", x);
        let encfile = format!("data/y_test/{}.enc",i);
        println!("write {}",encfile);
        let c = VectorLWE::encode_encrypt(&sk0, &y, &enc).unwrap();
        c.save(&encfile).unwrap();
    } 
    */
  
    Ok(())
}
