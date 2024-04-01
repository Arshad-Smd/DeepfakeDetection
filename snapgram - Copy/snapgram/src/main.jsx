import { BrowserRouter } from "react-router-dom";
import ReactDOM from "react-dom/client";
import App from "./App";
import { PostsProvider } from "./_root/pages/content";

ReactDOM.createRoot(document.getElementById("root")).render(
  <BrowserRouter>
    <PostsProvider>
      <App />
    </PostsProvider>
  </BrowserRouter>
);
