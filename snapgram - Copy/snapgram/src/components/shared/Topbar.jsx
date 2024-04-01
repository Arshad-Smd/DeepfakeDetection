
import { Link } from "react-router-dom";

import { Button } from "../ui/button";

// Simulated user data
const user = {
  id: 123,
  imageUrl: "/assets/icons/profile-placeholder.svg",
};

// Simulated sign out function
const signOut = () => {
  console.log("Signing out...");
};

const Topbar = () => {
  
  return (
    <section className="topbar">
      <div className="flex-between py-4 px-5">
        <Link to="/" className="flex gap-3 items-center">
          <img
            src="public\images\logo.svg"
            alt="logo"
            width={130}
            height={325}
          />
        </Link>

        <div className="flex gap-4">
          {/* Simulated sign out button */}
          <Button
            variant="ghost"
            className="shad-button_ghost"
            onClick={() => signOut()}
          >
            <img src="/assets/icons/logout.svg" alt="logout" />
          </Button>
          
          {/* Link to profile with simulated user data */}
          <Link to={`/profile/${user.id}`} className="flex-center gap-3">
            <img
              src={user.imageUrl}
              alt="profile"
              className="h-8 w-8 rounded-full"
            />
          </Link>
        </div>
      </div>
    </section>
  );
};

export default Topbar;
