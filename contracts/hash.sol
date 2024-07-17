// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract BinaryHash {
    bytes32 private lastComputedHash;
 
    event HashComputed(string input, bytes32 hash);

    // Function to compute hash from a string input
    function computeHash(string memory inputData) public {
        bytes memory inputBytes = bytes(inputData);
        bytes32 hash = keccak256(inputBytes);
        lastComputedHash = hash;
        emit HashComputed(inputData, hash);
    }

    // Function to get the last computed hash
    function computedHash() public view returns (bytes32) {
        return lastComputedHash;
    }
}
