// this outer vite.config.json is only for prepending ./ in front of paths when you run `npm run build` to make the `dist` folder
// so that when you try to package the electron app with `npm run package` it doesn't try to find absolute paths and explode
import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    outDir: 'dist', // Specify the output directory for production build
  },
  base: './', // Ensure all asset paths are relative (prepends './' in generated index.html)
});
