---
title: Adventures in Prediction Market Arbitrage
pubDate: 2024-03-31
---
Adventures in Prediction Market Arbitrage

<br>In this post I want to talk about my adventures in (trying) to arbitrage prediction markets, namely Polymarket and Kalshi.

<br>To get started I had to write indexing logic to get all current open markets. This was easy enough, they both have APIs documented at at [here](https://docs.polymarket.com/#markets-2) and [here](https://trading-api.readme.io/reference/getmarkets-1)        . I used asyncio, aiohttp, and aiolimiter in python to write a simple indexer that stored both to separate json files. It was slightly more annoying for Kalshi because it has a cursor based pagination setup instead of an offset based one. This meant that I couldn't just build my futures list and run gather, I had to sequentially keep iterating on the cursor. I also ended up doing batches of requests on Polymarket because I didn't know how many pages to fetch, but this wasn't a big deal either because I end up fetching everything in about 7 seconds.

<br>So from here I needed to figure out what markets matched on each market, but I didn't want to look for exact matches because I might miss markets that were actually the same thing but just worded differently. This is where I reached for the TfidfVectorizer from scikit-learn to embed the titles of the markets into the vector space defined by the term frequency-inverse document frequency matrices. Tf-IDF is the product of term frequency and inverse document frequency, where:

<br>TF(t,d) = the number of times term t appears in document d/total number of terms in document d
<br>IDF(t,D) = log(the total number of documents in the corpus D/number of documents that mention term t)

<br>Note that d is an element of D, and the total number of d in D is often termed as N

<br>So once the titles are all embedded into the vector space, I used pairwise cosine similarity and tried a few different similarity thresholds before dialing in on 0.8 as a threshold that has enough flexibility to allow for markets with somewhat different wording but not so much that it brings in a lot that are actually different markets because they end at different times. This does take a manual verification step because you still sometimes get markets like "Will Starlink get an FAA contract this year?" paired with "Will Starlink get an FAA contract before May?". I found it easiest to assume the markets are the same and do the verification AFTER checking to see if there is an arbitrage.

<br>Once I'd found similar markets, it was time to start comparing prices. To realize an arb, you have to take Yes on one market and No on another. The goal is to find markets where the price of Yes + No is less than 1, because then you end up with a guaranteed profit on expiry. Here's the catch though, you have to add fees per share in too. Polymarket has no trading fees, but Kalshi does as documented [here](https://kalshi.com/docs/kalshi-fee-schedule.pdf). What you find is:

<br>fees = round up(0.07 x C x P x (1-P))
<br>P = the price of a contract in dollars (50 cents is 0.5)
<br>C = the number of contracts being traded
<br>round up = rounds to the next cent

<br>The plot of fees ends up looking like:
<br><img src="/blog-posts/first-post/feeplot.png" alt="Fee plot for Kalshi contracts" width="300" />

<br>Ultimately, whenever I found what appeared to be an arb there were very small arbs with very little liquidity available to trade it at market and I found the idea of checking this and manually checking for liquidity and putting on the arb to be not worth my time. Were I to continue with this, I would start placing orders where it would actually be worth it and buying the other leg when I got filled. At the moment, I don't really want to deal with their orderbook api or automating this so I won't be continuing. Maybe one day I will, but that day is not today. I'd also need to figure out a way to filter out markets that aren't actually the same, which would need to go beyond their titles down to their rules for settlement. If not for that, I might actually tackle this right now.