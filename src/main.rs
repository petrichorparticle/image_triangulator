
use image::{open, DynamicImage, Rgb, Pixel, GenericImageView, imageops::FilterType, ImageBuffer};
use std::path::Path;

// For KMeans
use linfa::prelude::*;
use linfa_clustering::KMeans;
use linfa_nn::distance::L2Dist;
use ndarray::prelude::*;
use rand::prelude::*;

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

fn save_array_as_image(pixel_ind_array: Array2<usize>, centroids: Vec<Rgb<u8>>, file_path: &str) {
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

fn main(){
    let image_dir = "images";
    let in_name = "test.jpg";
    let out_name = "out.png";
    let n_clusters: usize = 12;
    let img: DynamicImage = open(format!("{image_dir}/{in_name}")).unwrap();

    // Resize
    let new_width = 1000;
    let (width, height) = img.dimensions();
    let new_height = (height as f32 * (new_width as f32 / width as f32)) as u32;
    let resized_img = img.resize_exact(new_width, new_height, FilterType::Lanczos3);

    // Convert to pixel array and cluster
    let pixel_array = load_image_to_2d_array(resized_img);
    let (pixel_ind_array, centroids) = colour_cluster(pixel_array, n_clusters);

    // Save pixel array
    save_array_as_image(pixel_ind_array, centroids, format!("{image_dir}/{out_name}").as_str());

    println!("We're done here!");
}
