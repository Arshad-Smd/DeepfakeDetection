import React, { useContext } from "react";
import { PostsContext } from "./content";

const Home = () => {
  const { posts, updateLikes } = useContext(PostsContext);
  console.log(posts)

  const handleLike = (postId) => {
    updateLikes(postId);
  };

  return (
    <div
      className="w-full grid flex-col justify-center items-center h-full bg-gradient-to-b from-gray-800 to-gray-900 m-auto"
      style={{ overflowY: "scroll" }}
    >
      <h2 className="h3-bold md:h2-bold text-left w-full pt-7">Home Feed</h2>
      {posts
        .slice()
        .reverse()
        .map((post) => (
          <div
            key={post.id}
            className="glassmorphism rounded-lg p-4 my-4 flex flex-col justify-center h-full items-center"
            style={{
              width: "800px",
              height: "85%",
              backgroundColor: "rgba(0,0,0,0.8)",
            }}
          >
            <div
              className="flex flex-col items-start my-3 py-4"
              style={{ width: "90%" }}
            >
              <h2
                className="text-2xl font-bold text-white py-3"
                style={{ color: "#877EFF" }}
              >
                {post.title}
              </h2>
              <p className="text-gray-200">{post.description}</p>
            </div>
            {post.mediaPreview && (
              <div>
                {post.mediaPreview.startsWith("data:image/") && (
                  <img
                    src={post.mediaPreview}
                    alt={post.title}
                    style={{ width: "90%" }}
                  />
                )}
                {post.mediaPreview.startsWith("data:audio/") && (
                  <div className="py-4">
                    <audio controls>
                      <source src={post.mediaPreview} type="audio/mpeg" />
                    </audio>
                  </div>
                )}
                {post.mediaPreview.startsWith("data:video/") && (
                  <div className="py-4">
                    <video controls>
                      <source src={post.mediaPreview} type="video/mp4" />
                    </video>
                  </div>
                )}
              </div>
            )}

            <div className="flex items-start py-8" style={{ width: "90%" }}>
              <button
                onClick={() => handleLike(post.id)}
                className="mt-2 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 focus:outline-none focus:bg-blue-600"
                style={{ backgroundColor: "#877EFF" }}
              >
              { post.likes ? <p>Unlike</p> : <p>Like</p>}
              </button>
            </div>
          </div>
        ))}
    </div>
  );
};

export default Home;