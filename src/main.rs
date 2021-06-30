use std::{env, error::Error, fs::File};

use image::{
    gif::{GifEncoder, Repeat},
    imageops::{overlay, resize, FilterType},
    io::Reader,
    DynamicImage, Frame, GenericImageView, Pixel, Primitive, Rgb, RgbImage,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

const C: Rgb<u8> = Rgb([197, 17, 17]);
const C2: Rgb<u8> = Rgb([122, 8, 56]);

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let ty: u32 = args[1].parse()?;
    let file = &args[2];
    let bg = Reader::open("dumpy/black.png")?.decode()?;
    let input = Reader::open(file)?.decode()?;

    let txd = input.width() as f64 / input.height() as f64;
    let tx = (ty as f64 * txd * 0.862).round() as u32;

    let img = resize(&input, tx, ty, FilterType::Gaussian);

    let pad = 10;
    let ix = (tx * 74) + (pad * 2);
    let iy = (ty * 63) + (pad * 2);

    let frames = (0..6)
        .into_par_iter()
        .map(|index| {
            let mut frame = bg.resize_exact(ix, iy, FilterType::Gaussian);
            let mut count = index;
            let mut count2 = index;

            for y in 0..ty {
                for x in 0..tx {
                    let pixel_i = Reader::open(format!("dumpy/{}.png", count))
                        .unwrap()
                        .decode()
                        .unwrap();

                    let indexed_pixel_rgba = img.get_pixel(x, y); 
                    let mut indexed_pixel = indexed_pixel_rgba.to_rgb();
                    if indexed_pixel.0 == [255, 255, 255] {
                        indexed_pixel = Rgb([254, 254, 254]);
                    }

                    let pixel = shader(pixel_i, indexed_pixel);
                    overlay(&mut frame, &pixel, (x * 74) + pad, (y * 63) + pad);
                

                    count += 1;
                    if count == 6 {
                        count = 0;
                    }
                }

                count2 -= 1;
                if count2 == -1 {
                    count2 = 5;
                }
                count = count2;
            }

            let set_bg_colour =
                ImageShader::new(ColourMapper::new(Rgb([255, 255, 255]), Rgb([0, 2, 0])));
            let filtered_bg = set_bg_colour.filter(frame);

            Frame::new(filtered_bg.into_rgba8())
        })
        .collect::<Vec<_>>();

    let file_out = File::create("out.gif")?;
    let mut encoder = GifEncoder::new_with_speed(file_out, 30);
    encoder.set_repeat(Repeat::Infinite)?;
    encoder.encode_frames(frames)?;

    Ok(())
}

fn shader(t: DynamicImage, p_rgb: Rgb<u8>) -> DynamicImage {
    let hsv = rgb_to_hsv(p_rgb);
    let black_level = 0.2f32;
    let entry = if hsv[2] < black_level {
        hsv_to_rgb([hsv[0], hsv[1], black_level])
    } else {
        p_rgb
    };

    let shade = Rgb([
        (entry.0[0] as f64 * 0.66) as u8,
        (entry.0[1] as f64 * 0.66) as u8,
        (entry.0[2] as f64 * 0.66) as u8,
    ]);

    let mut hsv = rgb_to_hsv(shade);
    hsv[0] -= 0.0635f32;
    if hsv[0] < 0f32 {
        hsv[0] += 1f32;
    }

    let shade = hsv_to_rgb(hsv);
    let lookup = ImageShader::new(ColourMapper::new(C, entry));
    let lookup2 = ImageShader::new(ColourMapper::new(C2, shade));
    let converted = lookup.filter(t);

    lookup2.filter(converted)
}

struct ImageShader {
    colour_mapper: ColourMapper<u8>,
}

impl ImageShader {
    pub fn new(colour_mapper: ColourMapper<u8>) -> Self {
        Self { colour_mapper }
    }

    pub fn filter(&self, image: DynamicImage) -> DynamicImage {
        let img = image.to_rgb8();

        let pixels = img
            .pixels()
            .map(|x| self.colour_mapper.lookup_pixel(x))
            .flat_map(|x| IntoIterator::into_iter(x.0));

        let img = RgbImage::from_vec(image.width(), image.height(), pixels.collect()).unwrap();

        DynamicImage::ImageRgb8(img)
    }
}

struct ColourMapper<T>
where
    T: Primitive,
{
    from: Rgb<T>,
    to: Rgb<T>,
}

impl<T: Primitive> ColourMapper<T> {
    pub fn new(from: Rgb<T>, to: Rgb<T>) -> Self {
        ColourMapper { from, to }
    }

    pub fn lookup_pixel(&self, src: &Rgb<T>) -> Rgb<T> {
        if src == &self.from {
            self.to
        } else {
            *src
        }
    }
}

fn rgb_to_hsv(input: Rgb<u8>) -> [f32; 3] {
    let [r, g, b] = input.0;
    let maxc = input.0.iter().max().unwrap();
    let minc = input.0.iter().min().unwrap();
    let v = maxc;

    if minc == maxc {
        return [0f32, 0f32, (*v as f32) / 255f32];
    }

    let diffc = (maxc - minc) as f32;

    let s = diffc / (*maxc as f32);
    let rc = (maxc - r) as f32 / diffc;
    let gc = (maxc - g) as f32 / diffc;
    let bc = (maxc - b) as f32 / diffc;
    let mut h = if &r == maxc {
        bc - gc
    } else if &g == maxc {
        2f32 + rc - bc
    } else {
        4f32 + gc - rc
    };

    h = (h / 6f32) % 1f32;

    [h, s, (*v as f32) / 255f32]
}

fn hsv_to_rgb(hsv: [f32; 3]) -> Rgb<u8> {
    let [h, s, v] = hsv;
    let return_v = (v * 255f32) as u8;
    if s == 0f32 {
        return Rgb([return_v, return_v, return_v]);
    }

    let i = (h * 6f32) as u8;
    let f = (h * 6f32).fract();
    let p = return_v as f32 * (1f32 - s);
    let q = return_v as f32 * (1f32 - s * f);
    let t = return_v as f32 * (1f32 - s * (1.0 - f));
    let i_mod = i % 6;
    match i_mod {
        0 => Rgb([return_v, t as u8, p as u8]),
        1 => Rgb([q as u8, return_v, p as u8]),
        2 => Rgb([p as u8, return_v, t as u8]),
        3 => Rgb([p as u8, q as u8, return_v]),
        4 => Rgb([t as u8, p as u8, return_v]),
        5 => Rgb([return_v, p as u8, q as u8]),
        _ => unreachable!(),
    }
}
