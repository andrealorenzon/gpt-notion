# gpt-notion


## installation

### requirements: 

#### git+bash

https://gitforwindows.org/


#### uv

https://docs.astral.sh/uv/getting-started/installation/


### download 

`git clone https://github.com/andrealorenzon/gpt-notion.git`

`cd gpt-notion`

edit `.env` with your API keys

### setup database on Supabase

* Head over to https://database.new to provision your Supabase database.
* In the studio, jump to the SQL editor and run the following script to enable pgvector and setup your database as a vector store:
```sql
-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Create a table to store your documents
create table
  documents (
    id uuid primary key,
    content text, -- corresponds to Document.pageContent
    metadata jsonb, -- corresponds to Document.metadata
    embedding vector (1536) -- 1536 works for OpenAI embeddings, change as needed
  );

-- Create a function to search for documents
create function match_documents (
  query_embedding vector (1536),
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
) language plpgsql as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding;
end;
$$;
```


`uv run main.py --query "YOUR QUESTION HERE" [--model "gpt-3.5-turbo"]`


