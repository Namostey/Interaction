const Web3 = require('web3');
const fs = require('fs');

const contractABI = require('./build/contracts/BinaryHash.json').abi;

// Configure your Web3 provider
const web3 = new Web3(new Web3.providers.HttpProvider('http://localhost:7545')); // Assuming Ganache is running on localhost:7545

// Replace with your deployed contract address and your account address
const contractAddress = '0x62Edd39aeEA835f0410Ab5B4b25529FF259dB471';
const fromAddress = '0x37ACB08A530DdAf3577cCE40c580976Df5515b3d';

// Read hash from file (assuming it's stored in hash_output.txt by Python script)
const hashFromFile = fs.readFileSync('hash_output.txt', 'utf8').trim();

// Instantiate the contract
const contract = new web3.eth.Contract(contractABI, contractAddress);

// Function to interact with Ethereum smart contract
async function interactWithSmartContract(dataHash) {
    try {
        // Encode the function call
        const encodedData = contract.methods.computeHash(dataHash).encodeABI();

        // Estimate gas cost
        const gas = await web3.eth.estimateGas({
            to: contractAddress,
            data: encodedData,
            from: fromAddress,
        });

        // Get gas price
        const gasPrice = await web3.eth.getGasPrice();

        // Send transaction
        const tx = await web3.eth.sendTransaction({
            to: contractAddress,
            data: encodedData,
            gas,
            gasPrice,
            from: fromAddress,
        });

        console.log('Transaction sent:', tx);
    } catch (error) {
        console.error('Error sending transaction:', error);
    }
}

// Example usage: Call interactWithSmartContract function
interactWithSmartContract(hashFromFile);
