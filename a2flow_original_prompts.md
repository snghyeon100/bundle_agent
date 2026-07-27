# A²Flow Original Prompts

## Case-based Initial Operators Generation Prompt

````python
GEN_CASE_OPS_PROMPT="""
You are a workflow design expert. The current {task_dtype} task is to extract workflow-related execution operations (execution operators) from specific problem descriptions and task resolutions. These execution operators are LLM-driven execution units.
- The operator’s input includes contextual history and a functional description of the operator. The execution of this operator is ultimately implemented through the LLM (which may be supplemented by external tools).
- Extract key workflow nodes (e.g., execution, validation, etc.) that can be applied to similar tasks across different workflows.

The specific use case is as follows:
class Custom(object):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        self.llm = llm
        self.operator_prompt = "This is a common chat operator. "
    async def __call__(self, input):
        response = await self.llm(input, self.operator_prompt)
        return response

The final output format should be:
```python
class AsyncLLM:
    async def __call__(self, input, prompt):
        # Simulate an LLM response
        return "Processed: input with prompt: prompt"

class [operator_name](object):
    def __init__(self, llm: AsyncLLM=AsyncLLM(), name: str = "operator_name"):
        self.llm = llm
        self.operator_prompt = "[operator_prompt]" # only need string type’s description for the operator function
        # self.tool = tool_func() if the operator needs a tool.
    async def __call__(self, input):
        response = await self.llm(input, self.operator_prompt)
        # if the operator need tools, add the self.tool_func()
        # response = self.tool(response)
        return response

class workflow(object):
    def __init__(self,):
        self.llm = AsyncLLM()
        self.operator_list = [operator1, operator2, operator3, ...]

    async def __call__(self, problem):
        history = problem # cannot be adjusted
        for operator in self.operator_list[:-1]:
            history += await operator(history)
        reponse = await self.operator_list[-1](history)
        return reponse
```
"""
````

## Operator Clustering and Preliminary Abstraction Prompt

````python
CLUSTER_OPS_PROMPT="""
Multiple task workflows contain execution operators. To derive clustered operators, you must:
- Cluster related/similar operators and prune non-essential ones
- Abstract generalized execution operations (e.g., execution, validation, etc.)
- Ensure function names are concise, non-redundant, and minimize total operator count
- Express the solution as Python code
The specific use case is as follows:
class Custom(object):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        self.llm = llm
        self.operator_prompt = "This is a common chat operator. "
    async def __call__(self, input):
        response = await self.llm(input, self.operator_promp)
        return response

The final output format is:
```python
# the AsyncLLM code cannot be adjusted for simulation
class AsyncLLM:
    async def __call__(self, input, prompt):
        # Simulate an LLM response
        return "Processed: input with prompt: prompt"

class [operator_name](object):
    def __init__(self, llm: AsyncLLM, name: str = "operator_name"):
        self.llm = llm
        self.operator_prompt = "[operator_prompt]" # only need string type’s description for the operator function
        # self.tool = tool_func() if the operator needs a tool.
    async def __call__(self, input):
        response = await self.llm(input, self.operator_prompt)
        \""" if the operator need tools \"""
        # response = self.tool(response)
        return response
```
"""
````

## Deep Extraction for Abstract Execution Operators Prompt

````python
DEEP_OPS_PROMPT="""
The input is some operations, but it is not abstract and general enough.
- Deep merge and generate general execution operations (e.g., execution, validation, etc.), and is not more than two words as operation name.
- Similar execution operations need to be merged to reduce the number of sub-operations.
- It is required to be expressed in the form of Python code function.

The specific use case is as follows:
class Custom(object):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        self.llm = llm
        self.operator_prompt = "This is a common chat operator. "

    async def __call__(self, input):
        response = await self.llm(input, self.operator_promp)
        return response

The final output format is:
```python
# the AsyncLLM code cannot be adjusted for simulation
class AsyncLLM:
    async def __call__(self, input, prompt):
        # Simulate an LLM response
        return "Processed: input with prompt: prompt"

class [operator_name](object):
    def __init__(self, llm: AsyncLLM, name: str = "operator_name"):
        self.llm = llm
        self.operator_prompt = "[operator_prompt]" # only need string type’s description for the operator function
        # self.tool = tool_func() if the operator needs a tool.
    async def __call__(self, input):
        response = await self.llm(input, self.operator_promp)
        \""" if the operator need tools \"""
        # response = self.tool(response)
        return response
```
"""
````

## Deep Next Prompt

```python
DEEP_NEXT_PROMPT="Please determine if deep-merging can be continued and proceed to deep-merge and create general execution operations."
```

## Workflow Optimize Prompt

```python
WORKFLOW_OPTIMIZE_PROMPT = """
You are building a Graph and corresponding Prompt to jointly solve {type} problems. Referring to the given graph and prompt, which forms a basic example of a {type} solution approach, please reconstruct and optimize them. You can add, modify, or delete nodes, parameters, or prompts. Include your single modification in XML tags in your reply. Ensure they are complete and correct to avoid runtime failures. When optimizing, you can incorporate critical thinking methods like review, revise, ensemble (generating multiple answers through different/similar prompts, then voting/integrating/checking the majority to obtain a final answer), selfAsk, etc. Consider Python’s loops (for, while, list comprehensions), conditional statements (if-elif-else, ternary operators), or machine learning techniques (e.g., linear regression, decision trees, neural networks, clustering). The graph complexity should not exceed 10. Use logical and control flow (IF-ELSE, loops) for a more enhanced graphical representation.Ensure that all the prompts required by the current graph from prompt_custom are included.Exclude any other prompts. Output the modified graph and all the necessary Prompts in prompt_custom (if needed). The prompt you need to generate is only the one used in ‘prompt_custom.XXX‘ within Custom. Other methods already have built-in prompts and are prohibited from being generated. Only generate those needed for use in ‘prompt_custom‘; please remove any unused prompts in prompt_custom. the generated prompt must not contain any placeholders. Considering information loss, complex graphs may yield better results, but insufficient information transmission can omit the solution. It’s crucial to include necessary context during the process."""
```

## Workflow Custom Use

````python
WORKFLOW_OPTIMIZE_PROMPT = """
Here’s two examples of using the "op1, op2" operators in graph:
the first example is sequence processing:
```
history = problem # the workflow’s memory
response = await self.op1(input=history)
history += "op1: " + response[’response’]
solution = await self.op2(input=history)
```
the second example is parallel solution mechanism:
```
history = problem # the workflow’s memory
for i in range(3):
    response = await self.op1(input=history)
    history += f"op1, iter {i}: " + response[’response’]
solution = await self.op2(input=history)
```
````

