const BinaryHash = artifacts.require('BinaryHash');

contract('BinaryHash', (accounts) => {
  it('should compute and store hash correctly', async () => {
    const instance = await BinaryHash.deployed();
    const input = 'test input';
    await instance.computeHash(input);
    const storedHash = await instance.computedHash();

    // Compute expected hash
    const expectedHash = web3.utils.keccak256(input);

    assert.equal(storedHash, expectedHash, 'Hash stored is not correct');
  });
});
