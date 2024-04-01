import { Outlet, Navigate } from "react-router-dom";

const AuthLayout = () => {
  const isAuthenticated = false;

  return (
    <>
      {isAuthenticated ? (
        <Navigate to="/" />
      ) : (
        <>
          <section className="flex flex-1 justify-center items-center flex-col py-10">
            <Outlet />
          </section>
          <img
            className="hidden xl:block w-1/2 h-screen object-cover bg-no-repeat"
            src="public\images\side-img.svg"
            alt="logo"
          />
        </>
      )}
    </>
  );
};

export default AuthLayout;
