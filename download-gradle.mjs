import https from 'https';
import fs from 'fs';
import path from 'path';

const url = 'https://services.gradle.org/distributions/gradle-8.11.1-bin.zip';
const targetPath = path.resolve('gradle-8.11.1-bin.zip');
const file = fs.createWriteStream(targetPath);

console.log('📦 Downloading Gradle binary directly to local disk...');
console.log('URL:', url);

function download(reqUrl) {
  https.get(reqUrl, (res) => {
    if (res.statusCode === 301 || res.statusCode === 302 || res.statusCode === 307) {
      console.log('Following redirect to:', res.headers.location);
      download(res.headers.location);
      return;
    }
    if (res.statusCode !== 200) {
      console.error(`HTTP ${res.statusCode} error fetching Gradle`);
      return;
    }

    const total = parseInt(res.headers['content-length'] || '0', 10);
    let downloaded = 0;
    let lastPct = -1;

    res.on('data', (chunk) => {
      downloaded += chunk.length;
      if (total > 0) {
        const pct = Math.floor((downloaded / total) * 100);
        if (pct % 10 === 0 && pct !== lastPct) {
          console.log(`Downloading: ${pct}% (${(downloaded / 1024 / 1024).toFixed(1)} MB / ${(total / 1024 / 1024).toFixed(1)} MB)`);
          lastPct = pct;
        }
      }
    });

    res.pipe(file);
    file.on('finish', () => {
      file.close();
      console.log('✅ Gradle 8.11.1 downloaded successfully to local disk!');
    });
  }).on('error', (err) => {
    console.error('Download error:', err.message);
  });
}

download(url);
