
use image::{open, DynamicImage, Rgb, Pixel, GenericImageView, imageops::FilterType, ImageBuffer};
use std::path::Path;

// For KMeans
use linfa::prelude::*;
use linfa_clustering::KMeans;
use linfa_nn::distance::L2Dist;
use ndarray::prelude::*;
use rand::prelude::*;
use std::cmp;

fn load_image_to_2d_array(img: DynamicImage)  -> Array3<u8> {

    let (width, height) = img.dimensions();

    // Create a 2D array of pixels
    let mut pixel_array = Array3::<u8>::zeros((width as usize,height as usize,3));

    for y in 0..height {
        for x in 0..width {
            // Get the pixel at the coordinates (x, y)
            let pixel = img.get_pixel(x, y).to_rgb();
            pixel_array.slice_mut(s![x as usize,y as usize,..]).assign(&arr1(&pixel.0));
        }
    }

    pixel_array
}

fn colour_cluster(pixel_array: Array3<u8>, n_clusters: usize) -> (Array2<usize>, Vec<Rgb<u8>>){
    let &[width,height, _] = pixel_array.shape() else { todo!() };

    // Go from a width x height array of rgb to a width*height x 3 array of f32
    let data: Array2<f32> = pixel_array.map(|&x| x as f32).into_shape((width*height,3))
        .expect("Something went wrong making the data the right shape.");
    let dataset = DatasetBase::from(data);

    // Perform KMeans clustering
    println!("Performing clustering!");
    let rng = thread_rng();
    let model = KMeans::params_with(n_clusters, rng, L2Dist)
        .max_n_iterations(200)
        .tolerance(1e-5)
        .fit(&dataset)
        .expect("Error while fitting KMeans to the dataset");

    // Convert centroids to u8
    let mut centroids: Vec<Rgb<u8>> = Vec::with_capacity(n_clusters);
    for k in 0..n_clusters {
        centroids.push(Rgb::from([model.centroids()[[k,0]] as u8,model.centroids()[[k,1]] as u8,model.centroids()[[k,2]] as u8]));
    }

    println!("Centroids are: {:?}", centroids);

    // Create pixel map
    let dataset = model.predict(dataset);
    let mut pixel_ind_array = Array2::<usize>::zeros((width as usize,height as usize));
    for x in 0..width {
        for y in 0..height {
            pixel_ind_array[[x,y]] = dataset.targets[y + x*height];
        }
    }

    (pixel_ind_array, centroids)
}

fn save_array_as_image(pixel_ind_array: &Array2<usize>, centroids: &Vec<Rgb<u8>>, file_path: &str) {
    let &[width,height] = pixel_ind_array.shape() else { todo!() };

    // Convert 2D array of indices into flat array of rgb values
    let mut flat_data = Vec::with_capacity(height * width * 3); // 3 bytes per pixel (RGB)
    for y in 0..height {
        for x in 0..width {
            flat_data.extend_from_slice(&centroids[pixel_ind_array[[x,y]]].0);
        }
    }

    // Create an ImageBuffer from the flattened data
    let buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(width as u32, height as u32, flat_data)
        .expect("Dimensions must match the flattened data length");

    // Save the image as a PNG file
    buffer.save(Path::new(file_path)).unwrap();
}

fn triangle_area(p1: (i32,i32), p2: (i32,i32), p3: (i32,i32)) -> i32 {
    ((p1.0 * (p2.1 - p3.1))
        + (p2.0 * (p3.1 - p1.1))
        + (p3.0 * (p1.1 - p2.1))).abs()
}

fn is_point_in_triangle(p: (i32,i32), corners: ((i32,i32),(i32,i32),(i32,i32))) -> bool {
    let (p1,p2,p3) = corners;
    let area = triangle_area(p1, p2, p3) as f64;

    let area1 = triangle_area(p, p2, p3) as f64;
    let area2 = triangle_area(p1, p, p3) as f64;
    let area3 = triangle_area(p1, p2, p) as f64;

    (area1 + area2 + area3 - area).abs() < 1e-6
}

fn triangle_score(target_pixels: &Array2<usize>, current_pixels: &Array2<usize>, corners: ((i32,i32),(i32,i32),(i32,i32)), colour: usize) -> (Array2<usize>, i32) {
    let ((x0,y0),(x1,y1),(x2,y2)) = corners;
    let max_x = cmp::max(cmp::max(x0,x1),x2);
    let min_x = cmp::min(cmp::min(x0,x1),x2);
    let max_y = cmp::max(cmp::max(y0,y1),y2);
    let min_y = cmp::min(cmp::min(y0,y1),y2);
    assert!(min_x >= 0);
    assert!(min_y >= 0);

    let mut score: i32 = 0;
    let mut new_pixels = current_pixels.clone();
    for x in min_x..(max_x+1) {
        for y in min_y..(max_y+1) {
            if is_point_in_triangle((x, y), corners) {
                let xu = x as usize;
                let yu = y as usize;
                new_pixels[[xu,yu]] = colour;
                if (colour == target_pixels[[xu,yu]]) && (colour != current_pixels[[xu,yu]]) {
                    score += 1;
                }
                else if (current_pixels[[xu,yu]] == target_pixels[[xu,yu]]) && (colour != target_pixels[[xu,yu]]) {
                    score -= 1;
                }
            }
        }
    }

    (new_pixels, score)
}

fn main(){
    let image_dir = "images";
    let in_name = "test.jpg";
    let out_name = "out";
    let out_ext = "png";
    let num_colours: usize = 12;
    let num_rounds: usize = 1000;
    let num_triangles_per_round: usize = 10000;
    let img: DynamicImage = open(format!("{image_dir}/{in_name}")).unwrap();

    // Resize
    let new_width = 1000;
    let (width, height) = img.dimensions();
    let new_height = (height as f32 * (new_width as f32 / width as f32)) as u32;
    let resized_img = img.resize_exact(new_width, new_height, FilterType::Lanczos3);

    // Convert to pixel array and cluster
    let pixel_array = load_image_to_2d_array(resized_img);
    let (target_pixels, centroids) = colour_cluster(pixel_array, num_colours);
    save_array_as_image(&target_pixels, &centroids, format!("{image_dir}/{out_name}_kmeans.{out_ext}").as_str());
    
    let mut art_pixels = Array2::<usize>::zeros(target_pixels.raw_dim());
    let mut rng = rand::thread_rng();
    let mut best_pixels = art_pixels.clone();
    let mut best_score: i32 = 0;
    for rounds in 0..num_rounds {
        save_array_as_image(&art_pixels, &centroids, format!("{image_dir}/{out_name}_{rounds}.{out_ext}").as_str());
        best_score = 0;
        for _ in 0..num_triangles_per_round {
            let p0: (i32,i32) = (rng.gen_range(0..new_width) as i32, rng.gen_range(0..new_height) as i32);
            let p1: (i32,i32) = (rng.gen_range(0..new_width) as i32, rng.gen_range(0..new_height) as i32);
            let p2: (i32,i32) = (rng.gen_range(0..new_width) as i32, rng.gen_range(0..new_height) as i32);
            let colour: usize = rng.gen_range(0..num_colours);
            let (pixels, score) = triangle_score(&target_pixels, &art_pixels, (p0,p1,p2), colour);
            if score > best_score {
                best_pixels = pixels;
                best_score = score;
            }
        }
        art_pixels = best_pixels.clone();
    }

    save_array_as_image(&art_pixels, &centroids, format!("{image_dir}/{out_name}_final.{out_ext}").as_str());

    println!("We're done here!");
}
