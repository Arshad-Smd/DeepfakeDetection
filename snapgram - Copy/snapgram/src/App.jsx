import { Route, Routes } from "react-router-dom";


import "./global.css";
import SigninForm from "./_auth/forms/SigninForm";
import { Home } from "./_root/pages";
import SignupForm from "./_auth/forms/SignupForm";
import AuthLayout from "./_auth/AuthLayout";
import RootLayout from "./_root/RootLayout";
import CreatePost from "./_root/pages/CreatePost";

const App = () => {
  return (
    <main className="flex h-screen">
      <Routes>
        <Route element={<AuthLayout />}>
          <Route path="/sign-in" element={<SigninForm />} />
          <Route path="/sign-up" element={<SignupForm />} />
        </Route>

        <Route element={<RootLayout />}>
          <Route path="/" index element={<Home />} />
          <Route path="/create-post" index element={<CreatePost />} />
        </Route>
      </Routes>
    </main>
  );
};

export default App;
