// PostsContext.js
import { createContext, useState } from "react";

export const PostsContext = createContext();

export const PostsProvider = ({ children }) => {
  const [posts, setPosts] = useState([]);

  const updateLikes = (postId) => {
    setPosts((prevPosts) =>
      prevPosts.map((post) =>
        post.id === postId ? { ...post, likes: post.likes === 0 ? 1 : 0 } : post
      )
    );
  };

  const addPost = (newPost) => {
    console.log(newPost);
    setPosts((prevPosts) => [...prevPosts, newPost]);
  };

  return (
    <PostsContext.Provider value={{ posts, addPost, updateLikes }}>
      {children}
    </PostsContext.Provider>
  );
};
