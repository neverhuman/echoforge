import { createRoot } from 'react-dom/client';
import App from './App';
import './styles.css';

const root = document.querySelector('#app');

if (!root) {
  throw new Error('EchoForge web root element #app was not found.');
}

createRoot(root).render(<App />);
