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
    //let N = x.nb_ciphertexts;
    //println!("lengths {:?}", N);

    let min = x.encoders[0].get_min() as f64;
    //println!("min {:?}", min);

    //let max = x.encoders[0].get_max() as f64;
    //println!("min {:?}", min);

    //let shift = -min*N as f64;
    let delta = x.encoders[0].delta as f64;

    //check that all of x have same encoder for each i
    //for i in 0..N {
    //    assert_eq!(x.encoders[i].get_min() as f64, min);
    //    assert_eq!(x.encoders[i].get_max() as f64, max);
    //}       
    //println!("same min and max for all");

    //println!("x_mod static");
    let x_sum = x.sum_with_new_min(min).unwrap();
    //stats(&x_sum); //x_mod.pp(); 

    let x_mod = x_sum.add_constant_static_encoder(&vec![0.9999f64 * delta]).unwrap();
    //stats(&x_mod); //x_mod.pp(); 

    return x_mod;
}

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
    let N = 197;
    
    // To execute sigmoid we need to set-up for bootstrapping
    
    //let enc = Encoder::new(-60., 60., 8, 0).unwrap();
    //let enc = Encoder::new(-30., 30., 8, 1)?;s
    let enc = Encoder::new(0.0, 1.0, 4, 1).unwrap();

    //println!("Load Bootstrapping Key 01 ... \n");
    //let bsk01_path = format!("{}/bsk01_LWE.json", path);
    //let bsk01 = LWEBSK::load(&bsk01_path);    

    fn sigmoid(v: f64) -> f64 {
      if v < -40.0 {
          0.0
      } else if v > 40.0 {
          1.0
      } else {
          1.0 / (1.0 + f64::exp(-1.*v))
      }
    }    
    
    /*
    // Function to sum a Vectopr LWE ciphertexts that might have negative values
    fn sum_ct_vec(mut c: VectorLWE, new_min: f64) -> VectorLWE{
        
        let lenght = c.nb_ciphertexts;
        let mut ct_min = 0.;
        let mut min = 0.;
        let mut ct_min_arr = vec![0.; lenght];
        
        for i in 0..lenght{
            min = f64::abs(f64::min(0., c.encoders[i].get_min() as f64));
            ct_min += min;
            ct_min_arr[i] = min;
        }
        
        c.add_constant_static_encoder_inplace(&ct_min_arr).unwrap();
        let mut ct = c.sum_with_new_min(ct_min+new_min).unwrap();
        ct.add_constant_dynamic_encoder_inplace(&[-1.*ct_min]).unwrap();
        
        return ct;
    }
    */
    
    // This is only for debugging
    println!("DEBUG: Load LWE secret keys ... \n");
    let sk0_LWE_path = format!("{}/sk0_LWE.json",path);
    let sk0 = LWESecretKey::load(&sk0_LWE_path).unwrap();    
    let sk1_LWE_path = format!("{}/sk1_LWE.json",path);
    let sk1 = LWESecretKey::load(&sk1_LWE_path).unwrap();   
    
    let sk1_RLWE_path = format!("{}/sk1_RLWE.json",path);
    let sk1_rlwe = RLWESecretKey::load(&sk1_RLWE_path).unwrap(); 
    
    let bsk01 = LWEBSK::new(&sk0, &sk1_rlwe, 6, 4);
    
    println!("DEBUG: Load ground truth data ... \n");
    let file = File::open("data/y_test.csv").expect("could not read data file");
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let y_test: Array2<f64> = reader.deserialize_array2_dynamic().expect("conversion error");
    //let N = y_test.shape()[0];
    // **************************
    
    println!("Number of data rows: {}", N);
    
    for i in 0..N {        
        let encfile = format!("data/X_test1/{}.enc",i);
        println!("{}", encfile);
        
        let features = VectorLWE::load(&encfile).unwrap();
        let terms = features.mul_constant_with_padding(&coeff, max_constant, nb_bit_padding)?;
        //let temps = terms.add_constant_static_encoder(&offset)?; 
        //let price = temps.sum_with_new_min(0.).unwrap();
        //let temps = terms.sum_with_new_min(-30.).unwrap(); 
        //let temps = terms.sum_with_padding().unwrap();
        
        /*let a = features.decrypt_decode(&sk0).unwrap();
        let b = coeff.clone();
        let p: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        println!("p = {}", p);*/
        
        //let temps = sum_ct_vec(terms, -30.);
        let temps = sum_vector_with_static_encoder(&terms);
        let z_pred = temps.add_constant_dynamic_encoder(&intercept)?;
        //z_pred.pp();
        
        // we actually could stop here and pass the linear prediction z
        // but for the purpose of illustration we next calculate sigmoid
        // to project the prediction on the interval 0. to 1.
        
        let y_pred = z_pred.bootstrap_nth_with_function(&bsk01, |x| sigmoid(x), &enc, 0)?;
        //let y_pred = z_pred.bootstrap_nth_with_function(&bsk01, |x| 1./(1. + f64::exp(x)), &enc, 0).unwrap();
        //let y_pred = z_pred.bootstrap_nth_with_function(&bsk01, |x| x*x/10., &enc, 0).unwrap();
        //let y_pred = z_pred.bootstrap_nth_with_function(&bsk01, |x| 1./(1.+x*x), &enc, 0)?;
        //let y_pred = z_pred.bootstrap_nth_with_function(&bsk01, |x| -x, &enc, 0)?;
        
        // This is only for debugging
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
            println!("DEBUG: y_pred {:?}", y_pred.decrypt_decode(&sk1).unwrap()); 
            //price.pp();
            println!("DEBUG: y_test {:?}", y_test.row(i).to_vec());
        } else if true {
            break;
        };        
        // **************************

        let encfile = format!("data/y_test1/{}.enc",i);
        y_pred.save(&encfile).unwrap();
    };
    
    Ok(())
}