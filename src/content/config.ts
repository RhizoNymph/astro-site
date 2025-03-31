import { defineCollection, z } from 'astro:content';

const blogCollection = defineCollection({
  type: 'content', //can be 'content' or 'data'
  schema: z.object({
    title: z.string(),
    pubDate: z.date(),
    description: z.string().optional(),
    tags: z.array(z.string()).optional(),
    // Add other frontmatter fields as needed
  }),
});

export const collections = {
  'blog': blogCollection,
}; 