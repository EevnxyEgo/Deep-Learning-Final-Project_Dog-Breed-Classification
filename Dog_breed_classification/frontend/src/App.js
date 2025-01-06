import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Container,
  Typography,
  Button,
  Box,
  LinearProgress,
  Grid,
  Card,
  CardContent,
  Collapse,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import { styled } from '@mui/system';
import ImageIcon from '@mui/icons-material/Image';
import { Github, Linkedin } from 'lucide-react';
import './Dogs.css';

const Input = styled('input')({
  display: 'none',
});

const breedDescriptions = {
  Chihuahua: "Chihuahua adalah salah satu ras anjing yang paling kecil di dunia, tetapi memiliki kepribadian yang besar. Chihuahua terkenal akan ukuran tubuh mungil, telinga berdiri tegak, dan kepribadian yang hidup. Meskipun kecil, Chihuahua memiliki hati yang penuh semangat dan sangat setia kepada pemiliknya. Mereka seringkali menjadi anjing pendamping yang penuh cinta dan ceria, cocok untuk berbagai gaya hidup.",
  "Japanese spaniel": "Japanese Chin atau yang juga dikenal sebagai Spaniel Jepang merupakan jenis anjing yang memiliki ciri khas yang dapat dilihat dari matanya yang juling. Ini adalah jenis anjing kuno yang berasal dari sekitar 700 M, dan populer di kalangan bangsawan Jepang. Kebanyakan pemilik menggambarkannya seperti kucing karena menggunakan cakarnya untuk membersihkan wajahnya. Ia juga suka melakukan trik seperti menari dengan kaki belakangnya sambil bertepuk tangan dengan kaki depannya.Jenis anjing ini bisa hidup antara 10-12 tahun, dan memiliki karakter yang waspada, cerdas, serta setia. Dia memiliki beberapa macam varian warna bulu mulai dari hitam dan putih.",
  "Maltese dog": "Maltese adalah sejenis anjing kecil dalam kategori anjing mainan. Nama anjing ini yang berarti 'dari Malta' dalam bahasa Inggris, biasanya tidak diterjemahkan dalam bahasa Indonesia.[2] Salah satu ciri khas anjing Maltese ialah bahwa bulunya tidak rontok, bulunya lembut seperti sutra. Keturunan anjing ini berasal dari Kawasan Mediterania Tengah. Nama Maltese ini berasal dari pulau Mediterania pulau Malta, tetapi, nama ini teradang diartikan dengan mengacu ke pulau Adriatic, atau sebuah kota mati bernama Melita Sicilian.",
  Pekinese: "Pekines atau Anjing Peking adalah ras anjing berukuran relatif kecil yang berasal dari Dinasti Tang, Cina pada abad ke-8.Penelitian menggunakan DNA membuktikan bahwa anjing ini termasuk salah satu ras yang paling tua di dunia. Anjing ini dihormati sebagai lambang penyebaran agama Buddha dari Tibet ke Cina pada tahun 700 SM.",
  Tzu: " Shih Tzu merupakan salah satu ras anjing kecil dan lucu yang populer di dunia. Shih Tzu memiliki bulu yang lebat dan panjang. Meski kecil, anjing ini sangat lincah, percaya diri, menyenangkan, dan sikap berani sehingga membuatnya menjadi favorit di antara penggemar anjing kecil.  Shih Tzu adalah ras anjing kuno dan memiliki sejarah panjang sebagai anjing peliharaan para bangsawan. Bila dilatih dan dirawat dengan baik, Shih Tzu dapat menjadi teman yang baik. Ukurannya yang kecil membuat ras anjing ini ideal untuk apartemen dan rumah berukuran kecil.",
  "Blenheim spaniel": "King Charles Spaniel (juga dikenal dengan nama English Toy Spaniel) adalah ras anjing kecil berjenis spaniel. Pada tahun 1903, the Kennel Club menyatukan empat ras toy spaniel yang terpisah menjadi King Charles Spaniel. Jenis-jenis lain yang digabung dengan ras ini adalah Blenheim, Ruby, dan Prince Charles Spaniels, masing-masing dengan warnanya sendiri.",
  papillon: "Papillon adalah ras anjing berukuran kecil yang berasal dari Prancis dan Jepang (Migrasi) sejak abad ke-16.[1] Namun anjing ini menjadi trend ikon fashion di Jepang mulai tahun 70-an. Perbedaannya Papillion Jepang warna coklat mendominasi karena hampir seluruh warna bulunya coklat dan ada sedikit kombinasi warna putih pada bagian telinga, leher dan dada,ujung ke empat kakinya dan pada bagian bawah badannya juga pada bagian pantatnya ada bulu putih menjuntai serta dibagian ekornya.Sedangkan dari Prancis warna putih menjadi dominan utama dan diikuti campuran warna hitam dan coklat.Dulunya, Papillon hanya dimiliki oleh kaum bangsawan seperti Marie Antoinette dan Madame Pompadour sebagai hewan hiasan atau dekoratif yang menarik perhatian.[1] Anjing ini memiliki tinggi 20–28 cm dan berat 4-4,5 kg.",
  "toy terrier": "English Toy Terrier adalah anjing kecil berwarna hitam dan cokelat dengan rambut halus dengan telinga runcing dan tegak. Mereka sedikit lebih panjang daripada tingginya. Idealnya, English Toy Terrier dewasa berukuran 25-30cm dan berat sekitar 2,7-3,6kg.",
  "Rhodesian ridgeback": "Anjing Rhodesian Ridgeback adalah anjing besar, berwarna solid, berkembang biak aktif dengan rambut pendek yang memiliki rambut khas di sepanjang punggungnya. Kuat dan lincah, Rhodesian Ridgeback jantan dewasa berukuran 63-69cm dan berat 30-39kg. Rhodesian Ridgeback betina memiliki tinggi 61-66cm dan berat 30-39kg. Mereka bisa memiliki warna rambut dari gandum muda hingga gandum merah.",
  "Afghan hound": "Afghan Hound adalah anjing berukuran sedang hingga besar dengan rambut panjang seperti sutra. Jenis anjing ini memiliki penglihatan panorama dan sendi pinggul unik yang memungkinkannya mencapai kecepatan luar biasa. Jenis anjing ini memiliki beberapa warna, dan sebagian besar pemilik menggambarkannya sebagai anjing yang ceria dengan kecenderungan untuk melucu. Jenis anjing ini sangat cocok untuk anak-anak. Bahkan pelukis terkenal Picasso pernah memelihara jenis anjing ini. Jenis anjing ini dapat hidup antara 11-13 tahun.",
};

function Project1() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [breedsOpen, setBreedsOpen] = useState(false);
  const [breeds, setBreeds] = useState([]);

  useEffect(() => {
    const fetchBreeds = async () => {
      try {
        const response = await fetch('/class_names.txt');
        const text = await response.text();
        const breedsArray = text.split('\n').filter(Boolean);
        setBreeds(breedsArray);
      } catch (error) {
        console.error('Error loading breeds:', error);
      }
    };
    fetchBreeds();
  }, []);

  const handleFileInput = (event) => {
    setSelectedFile(event.target.files[0]);
    setPredictionResult(null);

    const reader = new FileReader();
    reader.onload = () => {
      setPreviewUrl(reader.result);
    };
    reader.readAsDataURL(event.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      alert('Please select an image.');
      return;
    }
    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPredictionResult(response.data);
    } catch (error) {
      console.error('There was an error making the request', error);
      alert('Error making prediction.');
    } finally {
      setLoading(false);
    }
  };

  const toggleBreedsList = () => {
    setBreedsOpen(!breedsOpen);
  };

  return (
    <Container maxWidth="md" style={{ paddingTop: '50px' }}>
      <Typography
        variant="h3"
        component="h1"
        gutterBottom
        align="center"
        sx={{ fontWeight: 'bold', color: '#4A90E2', marginBottom: '30px' }}
      >
        AI Dog Breed Classifier
      </Typography>

      <Grid container spacing={4} justifyContent="center" alignItems="flex-start">
        <Grid item xs={12} md={6}>
          <Box
            textAlign="center"
            sx={{
              padding: '20px',
              backgroundColor: '#f0f4f8',
              borderRadius: '10px',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
            }}
          >
            <label htmlFor="contained-button-file">
              <Input
                accept="image/*"
                id="contained-button-file"
                type="file"
                onChange={handleFileInput}
              />
              <Button
                variant="contained"
                component="span"
                startIcon={<ImageIcon />}
                sx={{
                  backgroundColor: '#4A90E2',
                  '&:hover': { backgroundColor: '#357ABD' },
                }}
              >
                Upload Image
              </Button>
            </label>
            {previewUrl && (
              <Box mt={2}>
                <img
                  src={previewUrl}
                  alt="Preview"
                  style={{ maxWidth: '100%', height: 'auto', borderRadius: '10px' }}
                />
              </Box>
            )}
            <Box mt={2}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleSubmit}
                disabled={loading}
                sx={{
                  fontWeight: 'bold',
                  backgroundColor: '#5CB85C',
                  '&:hover': { backgroundColor: '#4CAE4C' },
                }}
              >
                {loading ? 'Predicting...' : 'Predict'}
              </Button>
            </Box>
            {loading && <LinearProgress style={{ marginTop: '20px' }} />}

            <Box mt={2}>
              <Button
                variant="outlined"
                onClick={toggleBreedsList}
                sx={{
                  fontWeight: 'bold',
                  color: '#4A90E2',
                  borderColor: '#4A90E2',
                  '&:hover': { borderColor: '#357ABD', color: '#357ABD' },
                }}
              >
                {breedsOpen ? 'Hide Breeds' : 'Show Possible Breeds'}
              </Button>
              <Collapse in={breedsOpen}>
                <Box
                  mt={2}
                  sx={{
                    maxHeight: '200px',
                    overflowY: 'auto',
                    border: '1px solid #ccc',
                    borderRadius: '4px',
                    padding: '10px',
                  }}
                >
                  <List dense>
                    {breeds.map((breed, index) => (
                      <ListItem key={index}>
                        <ListItemText primary={breed} />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              </Collapse>
            </Box>
          </Box>
        </Grid>

        <Grid item xs={12} md={6}>
          {predictionResult && (
            <Card
              elevation={3}
              sx={{
                padding: '20px',
                backgroundColor: '#ffffff',
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
                borderRadius: '10px',
              }}
            >
              <CardContent style={{ maxHeight: 'none', overflow: 'visible' }}>
                <Typography
                  variant="h5"
                  component="h2"
                  gutterBottom
                  sx={{ fontWeight: 'bold', color: '#4A90E2' }}
                >
                  Prediction Result
                </Typography>
                <Typography variant="body1">
                  <strong>Predicted Breed:</strong> {predictionResult.predicted_breed}
                </Typography>
                <Typography variant="body1">
                  <strong>Confidence:</strong> {(predictionResult.confidence).toFixed(2)}%
                </Typography>
                <Typography mt={2} style={{ whiteSpace: 'normal', wordWrap: 'break-word' }}>
                  {breedDescriptions[predictionResult.predicted_breed]}
              </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>

      <footer style={{ marginTop: '50px', textAlign: 'center' }}>
        <div className="social-links">
          <a
            href="https://github.com/EevnxyEgo"
            target="_blank"
            rel="noopener noreferrer"
            style={{ margin: '0 10px' }}
          >
            <Github size={32} />
          </a>
          <a
            href="https://www.linkedin.com/in/"
            target="_blank"
            rel="noopener noreferrer"
            style={{ margin: '0 10px' }}
          >
            <Linkedin size={32} />
          </a>
        </div>
        <Typography variant="body2" color="textSecondary">
          © 2024 Dog Breed Classifier. All Rights Reserved.
        </Typography>
      </footer>
    </Container>
  );
}

export default Project1;
