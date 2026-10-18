# Type-Signature-Inference-Engineering

This is a model I built (and works) but I was not able to submit it in time for the competition.
Below is the following objectives and rules for the previous competition.

In Haskell, the type signature written above a function is a compact specification: what goes in, what comes out, which effects (I/O, state, parsing) are involved, and which constraints apply to polymorphic arguments.
Programmers usually write signatures by hand, and a compiler can infer many of
them, but only with the whole program, its imports and its library types.
This challenge asks a model to read one function's equations and the few
lines of code above it, and write the signature the author wrote. The model
must recover argument and result types from how values are used, which
domain types a project defines, and which monads and type classes it
relies on, all for repositories it has never seen.
Models must be trained from scratch on the provided data.
Given a top-level Haskell function's name, its definition and up to 12
lines of code above its signature, generate the type signature written by
the author.

2. Objective
   
For each test row, predict type: the type in the function's signature
name :: type, without the name and ::. Example: for
enqueue pid m = M.insertWith (flip (++)) pid [m]  
the author wrote enqueue :: ProcessId -> m -> Queues m -> Queues m, so the
answer is ProcessId -> m -> Queues m -> Queues m.

3. Public Data
   
File	Description
train.csv	17,192 functions from 392 repositories, with their types.
test.csv	2,235 functions from 161 other repositories.
repo_licenses.csv	Licence texts of the repositories whose code appears in the public files, not linked to repository IDs.
sample_submission.csv	Valid submission predicting IO () for every row.


train.csv / test.csv columns (train.csv adds type):
Column	Type	Meaning
example_id	string	TR_##### / TE_#####.
repo_id	string	Opaque repository ID (RP_ + 8 hex characters).
name	string	The function's name.
before	string	Up to 12 source lines directly above the signature (may include other signatures, comments, imports or data declarations).
definition	string	The function's equations, as written below the signature (up to 25 lines, including where clauses and guards).
type	string (train only)	The type from the author's signature, with whitespace normalised.


Rows. Functions come from small, permissively licensed public Haskell
repositories. A row needs a single-name top-level signature followed
directly by the function's equations, and a type of 3–120 characters. Files
copied between repositories are removed. Each exact type appears at most 20
times, so boilerplate types such as IO () do not dominate. A test source
file contributes at most one row, so no test row's context shows another test
row's answer. Comments are kept as written; links and e-mail addresses are
replaced by <url> and <email>.

4. Submission Format
   
CSV with exactly the columns example_id,type, in that order, one row per
test example_id. Each type is a non-empty string of at most 1,000
characters. Quote values that contain commas.
example_id,type  
TE_00001,"Int -> [a] -> [a]"  
TE_00002,Parser Expr  

5. Evaluation
   
Predicted and true types are split into tokens: identifiers (including
qualified names such as M.Map and primes), ->, =>, ::, brackets,
parentheses and commas. Whitespace is ignored. A row scores 1 if the token
sequences are identical. Otherwise it scores 0.5 × token F1: the overlap of
the two token multisets, as 2·common / (predicted + true tokens). The final
score is the mean over test rows. Range [0, 1]; higher is better.
A submission is rejected with an explicit validation error (no score) if the
columns are wrong or out of order, any ID is missing, duplicated or unknown,
or any value is missing, empty or longer than 1,000 characters.

6. Split and Leakage Controls
   
- Split unit: repository. No row of a test repository appears in
  training.
- A test source file contributes at most one row.
- Repository names, file paths and links are not published.
  
7. Not Allowed
   
- No pretrained models of any kind (no pretrained language models, code
  models, embeddings or tokenisers trained on outside data); models must be
  trained from scratch on the provided files.
- No external data: no Haskell compilers or type checkers, library
  documentation, Hackage or code search, other source code, and no data from
  other challenges.
- Predictions may use the unlabelled inputs of other test.csv rows, but no
  labels or external data. Lets build this using Python as a script
