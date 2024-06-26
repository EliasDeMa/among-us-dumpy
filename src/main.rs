use std::{env, error::Error, fs::File};

use image::{
    codecs::gif::{GifEncoder, Repeat},
    DynamicImage,
    Frame,
    imageops::{FilterType, overlay, resize}, io::Reader, Primitive, Rgba, RgbaImage,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

const C: Rgba<u8> = Rgba([197, 17, 17, 255]);
const C2: Rgba<u8> = Rgba([122, 8, 56, 255]);

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let ty: u32 = args[1].parse()?;
    let file = &args[2];
    let out = &args[3];
    let bg = Reader::open("dumpy/black.png")?.decode()?.to_rgba8();
    let input = Reader::open(file)?.decode()?.to_rgba8();

    let txd = input.width() as f64 / input.height() as f64;
    let tx = (ty as f64 * txd * 0.862).round() as u32;

    let img = resize(&input, tx, ty, FilterType::CatmullRom);

    let pad = 10;
    let ix = (tx * 74) + (pad * 2);
    let iy = (ty * 63) + (pad * 2);

    let pixel_images: Vec<_> = (0..6)
        .map(|index| Reader::open(format!("dumpy/{}.png", index))
            .unwrap()
            .decode()
            .unwrap()
            .to_rgba8())
        .collect();

    let frames = (0..6)
        .into_par_iter()
        .map(|frame_index| {
            let mut frame = DynamicImage::ImageRgba8(bg.clone()).resize_exact(ix, iy, FilterType::Nearest).to_rgba8();

            for y in 0..ty {
                let y_pad = ((y * 63) + pad).into();
                for x in 0..tx {
                    let pixel_index = (frame_index + x as i32 - y as i32).rem_euclid(6) as usize;
                    let pixel_i = pixel_images[pixel_index].clone();

                    let indexed_pixel_rgba = img.get_pixel(x, y);
                    let indexed_pixel = *indexed_pixel_rgba;

                    let pixel = shader(DynamicImage::ImageRgba8(pixel_i), indexed_pixel);
                    overlay(&mut frame, &pixel.to_rgba8(), ((x * 74) + pad).into(), y_pad);
                }
            }

            Frame::new(frame)
        })
        .collect::<Vec<_>>();

    let file_out = File::create(out)?;
    let mut encoder = GifEncoder::new_with_speed(file_out, 30);
    encoder.set_repeat(Repeat::Infinite)?;
    encoder.encode_frames(frames)?;

    Ok(())
}

fn shader(t: DynamicImage, p_rgba: Rgba<u8>) -> DynamicImage {
    let hsva = rgba_to_hsva(p_rgba);
    let black_level = 0.2f32;
    let entry = if hsva[2] < black_level {
        hsva_to_rgba([hsva[0], hsva[1], black_level, hsva[3]])
    } else {
        p_rgba
    };

    let shade = Rgba([
        (entry.0[0] as f64 * 0.66) as u8,
        (entry.0[1] as f64 * 0.66) as u8,
        (entry.0[2] as f64 * 0.66) as u8,
        entry.0[3],
    ]);

    let mut hsva = rgba_to_hsva(shade);
    hsva[0] -= 0.0635f32;
    if hsva[0] < 0f32 {
        hsva[0] += 1f32;
    }

    let shade = hsva_to_rgba(hsva);
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
        let img = image.to_rgba8();

        let pixels = img
            .pixels()
            .map(|x| self.colour_mapper.lookup_pixel(x))
            .flat_map(|x| IntoIterator::into_iter(x.0));

        let img = RgbaImage::from_vec(image.width(), image.height(), pixels.collect()).unwrap();

        DynamicImage::ImageRgba8(img)
    }
}

struct ColourMapper<T>
where
    T: Primitive,
{
    from: Rgba<T>,
    to: Rgba<T>,
}

impl<T: Primitive> ColourMapper<T> {
    pub fn new(from: Rgba<T>, to: Rgba<T>) -> Self {
        ColourMapper { from, to }
    }

    pub fn lookup_pixel(&self, src: &Rgba<T>) -> Rgba<T> {
        if src == &self.from {
            self.to
        } else {
            *src
        }
    }
}

fn rgba_to_hsva(input: Rgba<u8>) -> [f32; 4] {
    let [r, g, b, a] = input.0;
    let maxc = input.0[..3].iter().max().unwrap();
    let minc = input.0[..3].iter().min().unwrap();
    let v = maxc;

    if minc == maxc {
        return [0f32, 0f32, (*v as f32) / 255f32, (a as f32) / 255f32];
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

    [h, s, (*v as f32) / 255f32, (a as f32) / 255f32]
}

fn hsva_to_rgba(hsva: [f32; 4]) -> Rgba<u8> {
    let [h, s, v, a] = hsva;
    let return_v = (v * 255f32) as u8;
    let return_a = (a * 255f32) as u8;
    if s == 0f32 {
        return Rgba([return_v, return_v, return_v, return_a]);
    }

    let i = (h * 6f32) as u8;
    let f = (h * 6f32).fract();
    let p = return_v as f32 * (1f32 - s);
    let q = return_v as f32 * (1f32 - s * f);
    let t = return_v as f32 * (1f32 - s * (1.0 - f));
    let i_mod = i % 6;
    match i_mod {
        0 => Rgba([return_v, t as u8, p as u8, return_a]),
        1 => Rgba([q as u8, return_v, p as u8, return_a]),
        2 => Rgba([p as u8, return_v, t as u8, return_a]),
        3 => Rgba([p as u8, q as u8, return_v, return_a]),
        4 => Rgba([t as u8, p as u8, return_v, return_a]),
        5 => Rgba([return_v, p as u8, q as u8, return_a]),
        _ => unreachable!(),
    }
}