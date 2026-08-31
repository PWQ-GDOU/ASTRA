const { spawn } = require('child_process');
const { resolve } = require('path');

const VITE_DEV_SERVER_URL = 'http://localhost:5173';

const electronPath = require('electron');
const mainEntry = resolve(__dirname, '..', 'dist-electron', 'main', 'index.js');

console.log('Starting Electron...');
console.log('  electron:', electronPath);
console.log('  main entry:', mainEntry);
console.log('  dev server:', VITE_DEV_SERVER_URL);

const child = spawn(electronPath, [mainEntry], {
  env: {
    ...process.env,
    VITE_DEV_SERVER_URL,
    NODE_ENV: 'development',
  },
  stdio: 'inherit',
});

child.on('close', (code) => {
  process.exit(code);
});
