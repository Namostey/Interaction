const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const port = 5000;

app.use(bodyParser.json());

// Endpoint to receive data from the Ethereum contract
app.post('/hash', (req, res) => {
  const { input, hash } = req.body;
  console.log(`Received hash: ${hash} for input: ${input}`);
  // Perform actions with the received data
  res.send('Data received');
});

app.listen(port, () => {
  console.log(`Backend server listening at http://localhost:${port}`);
});
