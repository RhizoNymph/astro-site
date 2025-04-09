import { defineCollection, z } from 'astro:content';

// Define the schema for the blog collection
const blogCollection = defineCollection({
  type: 'content', // 'content' for markdown/mdx, 'data' for json/yaml
  schema: z.object({
    title: z.string(),
    pubDate: z.coerce.date(), // Coerce string dates into Date objects
    description: z.string().optional(), // Optional description
    // Add any other frontmatter fields you use
    // image: z.object({
    //   url: z.string(),
    //   alt: z.string()
    // }).optional(),
    // tags: z.array(z.string()).optional(),
  }),
});

// Export a single 'collections' object to register your collection(s)
export const collections = {
  'blog': blogCollection,
};