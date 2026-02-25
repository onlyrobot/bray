openssl req -x509 -newkey rsa:4096 -sha256 -days 3650 -nodes \
  -keyout key.pem -out cert.pem \
  -subj "/CN=47.84.5.93" \
  -addext "subjectAltName=DNS:47.84.5.93,IP:127.0.0.1"