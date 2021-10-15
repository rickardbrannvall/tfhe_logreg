#![allow(non_snake_case)]
use concrete::*;

fn main() -> Result<(), CryptoAPIError> {

    let path = "keys";
    
    println!("loading LWE key... \n");
    let sk0_LWE_path = format!("{}/sk0_LWE.json",path);
    let sk0 = LWESecretKey::load(&sk0_LWE_path).unwrap();    
    
    // create an encoder
    let enc = Encoder::new(0., 10., 6, 4)?;

    let m0: Vec<f64> = vec![2.54];
    println!("plaintext value {:?}\n", m0);
    
    let c0 = VectorLWE::encode_encrypt(&sk0, &m0, &enc)?;  
    println!("encrypted value {:?}", c0.decrypt_decode(&sk0).unwrap());
    c0.pp();
    
    let constants: Vec<f64> = vec![1.0];
    
    let mut ct = VectorLWE::encode_encrypt(&sk0, &m0, &enc)?;  
    ct.add_constant_static_encoder_inplace(&constants)?; 
    println!("add constant one {:?}", ct.decrypt_decode(&sk0).unwrap());
    ct.pp();   

    let mut ct = VectorLWE::encode_encrypt(&sk0, &m0, &enc)?;  
    ct.add_with_padding_inplace(&c0)?;
    println!("add with padding {:?}", ct.decrypt_decode(&sk0).unwrap());
    ct.pp();       

    ct = VectorLWE::encode_encrypt(&sk0, &m0, &enc)?;  
    ct.add_with_new_min_inplace(&c0, &vec![0.0])?;
    println!("add with new min {:?}", ct.decrypt_decode(&sk0).unwrap());
    ct.pp();     

    let max_constant: f64 = 1.0;
    let nb_bit_padding = 4;

    ct = VectorLWE::encode_encrypt(&sk0, &m0, &enc)?;  
    ct.mul_constant_with_padding_inplace(&constants, max_constant, nb_bit_padding)?;
    println!("mul constant one {:?}", ct.decrypt_decode(&sk0).unwrap());
    ct.pp();      

    ct = VectorLWE::encode_encrypt(&sk0, &m0, &enc)?;  
    ct.opposite_nth_inplace(0).unwrap();
    println!("negation of val {:?}", ct.decrypt_decode(&sk0).unwrap());
    ct.pp();      
    
    
    let zero: Vec<f64> = vec![0.0];
    let z_star = VectorLWE::encode_encrypt(&sk0, &zero, &enc)?;
    let encfile = "z_star.enc";
    z_star.save(&encfile).unwrap();
    
    let a_star = z_star.add_constant_static_encoder(&vec![3.0])?;
    let b_star = z_star.add_constant_static_encoder(&vec![5.0])?;
    
    let c_star = b_star.sub_with_padding(&a_star)?;
    println!("c {:?}", c_star.decrypt_decode(&sk0).unwrap());
    c_star.pp();      
    
    let encfile = "c_star.enc";
    c_star.save(&encfile).unwrap();

    let z_dagg = VectorLWE::encode_encrypt(&sk0, &zero, &enc)?;
    let encfile = "z_dagg.enc";
    z_dagg.save(&encfile).unwrap();

    let b_dagg = z_dagg.add_constant_static_encoder(&vec![5.0])?;

    let c_dagg = b_dagg.sub_with_padding(&a_star)?;
    println!("c {:?}", c_dagg.decrypt_decode(&sk0).unwrap());
    c_dagg.pp();      
    
    let encfile = "c_dagg.enc";
    c_dagg.save(&encfile).unwrap();
 
    Ok(())
}
