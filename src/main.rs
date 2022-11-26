use mnist::*;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

const CLASS : usize = 10;
const IMDIM : usize = 28;
const TRAIN : usize = 50_000;
const USE_TRAIN : usize = 10_000;
const TEST  : usize = 10_000;
const RATE  : f64 = 0.05;

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit    ()
        .training_set_length   (TRAIN as u32)
        .validation_set_length (TEST as u32)
        .test_set_length       (TEST as u32)
        .finalize              ();

    let mut weight3d: Array3<f64> = Array3::random((CLASS, IMDIM, IMDIM), Uniform::new(-0.5, 0.5));
    for mut w in weight3d.outer_iter_mut() { // normalization
        let sum = w.sum();
        w.mapv_inplace(|a| a/sum);
    }

    let train_data: Array3<f64> = Array3::from_shape_vec((TRAIN, IMDIM, IMDIM), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.0);

    let train_labels: Array2<usize> = Array2::from_shape_vec((TRAIN, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as usize);

    for i in 0..USE_TRAIN {
        let label = train_labels[[i, 0]];
        let iact = train_data.slice(s![i,..,..]);
        let mut nnout : [f64; CLASS] = [0.; CLASS];
        for j in 0..CLASS {
            let w = weight3d.slice(s![j,..,..]);
            let mul = &iact * &w;
            nnout[j] = mul.sum();
        }

        // prediction result check
        let (pred,..) = nnout.iter().enumerate().fold((usize::MIN, f64::MIN), |(ia, a), (ib, &b)| {
            if b > a {
                (ib, b)
            } else {
                (ia, a)
            }
        });

        let tvec = Array::from_vec(
            (0..CLASS).map(|x| {if x == label {1.} else {0.}}).collect::<Vec<f64>>());
        let pvec = Array::from_vec(
            (0..CLASS).map(|x| {if x == pred {1.} else {0.}}).collect::<Vec<f64>>());
        let diff = (tvec - pvec) * RATE;

        for (idx, mut w) in weight3d.outer_iter_mut().enumerate() {
            let new_w = &w + &iact * diff[idx];
            w.assign(&new_w);
        }
    }


    // test set
    let test_data = Array3::from_shape_vec((TEST, IMDIM, IMDIM), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.);

    let test_labels: Array2<usize> = Array2::from_shape_vec((TEST, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as usize);

    // test
    let mut confmat : Array2<i32> = Array2::zeros((CLASS, CLASS)); // (pred, label)
    let mut cnt = 0;
    for (i, iact) in test_data.outer_iter().enumerate() {
        let label = test_labels[[i, 0]];
        let mut nnout : [f64; CLASS] = [0.; CLASS];
        for j in 0..CLASS {
            let w = weight3d.slice(s![j,..,..]);
            let mul = &iact * &w;
            nnout[j] = mul.sum();
        }

        // prediction result check
        let (pred,..) = nnout.iter().enumerate().fold((usize::MIN, f64::MIN), |(ia, a), (ib, &b)| {
            if b > a {
                (ib, b)
            } else {
                (ia, a)
            }
        });

        if &pred == &label {
            cnt += 1;
        }
        confmat[[pred as usize, label as usize]] += 1;
    }

    println!("Confusion Matrix");
    for ln in confmat.outer_iter() {
        print!("| ");
        for x in ln.iter() {
            print!("{:>4} ", x);
        }
        println!("|");
    }

    let rate = (cnt as f64) / (TEST as f64) * 100.0;
    println!("Prediction correct: {}/{} ({}%)", cnt, TEST, rate);
}
