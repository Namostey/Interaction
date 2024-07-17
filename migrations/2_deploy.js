const BinaryHash = artifacts.require("BinaryHash");

module.exports = function(deployer) {
  deployer.deploy(BinaryHash);
};
