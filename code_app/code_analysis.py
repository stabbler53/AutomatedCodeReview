import json
import tree_sitter_python
from tree_sitter import Language, Parser
import re
import logging
from typing import List, Dict, Any, Tuple

# Initialize the parser with Python language
PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser()
parser.set_language(PY_LANGUAGE)  # Use the language setter instead of the deprecated method

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAX_NESTING_DEPTH = 3
DANGEROUS_FUNCTIONS = {'eval', 'exec'}

def check_excessive_nesting(node: Any, max_depth: int = MAX_NESTING_DEPTH) -> bool:
    """
    Check if the code has excessive nesting.

    Parameters:
    node (Any): The root node of the code to check.
    max_depth (int): The maximum allowed depth of nesting.

    Returns:
    bool: True if the nesting exceeds max_depth, False otherwise.
    """
    def get_depth(n: Any, depth: int = 0) -> int:
        if n.type == 'block':
            depth += 1
        if depth > max_depth:
            return depth
        for child in n.children:
            depth = max(depth, get_depth(child, depth))
        return depth
    return get_depth(node) > max_depth

def has_docstring(node: Any) -> bool:
    """
    Check if a function has a docstring.

    Parameters:
    node (Any): The function node to check.

    Returns:
    bool: True if the function has a docstring, False otherwise.
    """
    if node.type != 'function_definition':
        return False
    docstring_node = node.child_by_field_name('body').child(0)
    return docstring_node and docstring_node.type == 'expression_statement' and docstring_node.child(0).type == 'string'

def calculate_cyclomatic_complexity(node: Any) -> int:
    """
    Calculate the cyclomatic complexity of a function.

    Parameters:
    node (Any): The function node to analyze.

    Returns:
    int: The cyclomatic complexity of the function.
    """
    complexity = 1
    nodes_to_visit = [node]
    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        if current_node.type in ['if_statement', 'for_statement', 'while_statement', 'try_statement', 'with_statement']:
            complexity += 1
        nodes_to_visit.extend(current_node.children)
    return complexity

def is_snake_case(name: str) -> bool:
    """
    Check if a name follows snake_case naming convention.

    Parameters:
    name (str): The name to check.

    Returns:
    bool: True if the name follows snake_case, False otherwise.
    """
    return re.match(r'^[a-z_][a-z0-9_]*$', name) is not None

def has_comment(node: Any) -> bool:
    """
    Check if a function has inline comments.

    Parameters:
    node (Any): The function node to check.

    Returns:
    bool: True if the function has inline comments, False otherwise.
    """
    nodes_to_visit = [node]
    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        if current_node.type == 'comment':
            return True
        nodes_to_visit.extend(current_node.children)
    return False

def contains_dangerous_functions(node: Any) -> bool:
    """
    Check if a function contains dangerous functions like eval or exec.

    Parameters:
    node (Any): The function node to check.

    Returns:
    bool: True if the function contains dangerous functions, False otherwise.
    """
    nodes_to_visit = [node]
    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        if current_node.type == 'call':
            function_node = current_node.child_by_field_name('function')
            if function_node and function_node.text.decode('utf8') in DANGEROUS_FUNCTIONS:
                return True
        nodes_to_visit.extend(current_node.children)
    return False

def generate_ast(node: Any) -> Dict[str, Any]:
    """
    Generate an abstract syntax tree (AST) from a given node.

    Parameters:
    node (Any): The root node of the code.

    Returns:
    Dict[str, Any]: The AST represented as a nested dictionary.
    """
    def node_to_dict(n: Any) -> Dict[str, Any]:
        return {
            'type': n.type,
            'children': [node_to_dict(child) for child in n.children]
        }
    return node_to_dict(node)

def analyze_control_flow_and_dependencies(node: Any) -> Tuple[List[str], List[str]]:
    """
    Analyze the control flow and data dependencies in the code.

    Parameters:
    node (Any): The root node of the code.

    Returns:
    Tuple[List[str], List[str]]: A tuple containing lists of control flow statements and data dependencies.
    """
    control_flow_statements = []
    data_dependencies = []
    nodes_to_visit = [node]

    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        if current_node.type in ['if_statement', 'for_statement', 'while_statement', 'try_statement', 'with_statement']:
            control_flow_statements.append(current_node.type)
        if current_node.type in ['identifier', 'assignment']:
            data_dependencies.append(current_node.text.decode('utf8'))
        nodes_to_visit.extend(current_node.children)

    return control_flow_statements, data_dependencies

def analyze_code(code: str) -> List[Dict[str, Any]]:
    """
    Analyze the given code for various quality metrics.

    Parameters:
    code (str): The code to analyze.

    Returns:
    List[Dict[str, Any]]: A list of findings for each function in the code.
    """
    # Parse the code
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    query = PY_LANGUAGE.query("""
    (function_definition
      name: (identifier) @function.def
      body: (block) @function.block)
    """)

    captures = query.captures(root_node)

    findings = []

    for capture in captures:
        node, capture_name = capture
        if capture_name == 'function.def':
            function_name = node.text.decode('utf8')
            function_body_node = node.parent.child_by_field_name('body')
            function_findings = {
                'function_name': function_name,
                'start_point': node.start_point,
                'issues': [],
                'ast': generate_ast(node.parent),
                'control_flow': [],
                'data_dependencies': []
            }

            # Check for excessive nesting
            if check_excessive_nesting(function_body_node):
                function_findings['issues'].append('Excessive nesting detected')

            # Check for missing docstrings
            if not has_docstring(node.parent):
                function_findings['issues'].append('Missing docstring')

            # Calculate cyclomatic complexity
            complexity = calculate_cyclomatic_complexity(function_body_node)
            function_findings['issues'].append(f'Cyclomatic complexity: {complexity}')

            # Check naming convention
            if not is_snake_case(function_name):
                function_findings['issues'].append('Does not follow snake_case naming convention')

            # Check for comments
            if not has_comment(function_body_node):
                function_findings['issues'].append('No inline comments')

            # Check for dangerous functions
            if contains_dangerous_functions(function_body_node):
                function_findings['issues'].append('Contains dangerous functions (e.g., eval, exec)')

            # Analyze control flow and data dependencies
            control_flow, data_dependencies = analyze_control_flow_and_dependencies(function_body_node)
            function_findings['control_flow'] = control_flow
            function_findings['data_dependencies'] = data_dependencies

            findings.append(function_findings)

    return findings

def evaluate_code_quality(findings: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Evaluate the overall code quality based on the findings.

    Parameters:
    findings (List[Dict[str, Any]]): The list of findings for each function.

    Returns:
    Dict[str, int]: A summary of the code quality metrics.
    """
    total_functions = len(findings)
    issues_count = sum(len(finding['issues']) for finding in findings)
    quality_score = max(0, 100 - issues_count * 10)
    return {
        'total_functions': total_functions,
        'total_issues': issues_count,
        'quality_score': quality_score
    }

def provide_feedback(findings: List[Dict[str, Any]]) -> List[str]:
    """
    Provide feedback and suggestions for improving the code based on the findings.

    Parameters:
    findings (List[Dict[str, Any]]): The list of findings for each function.

    Returns:
    List[str]: A list of feedback and suggestions for improvement.
    """
    feedback = []
    for finding in findings:
        function_name = finding['function_name']
        issues = finding['issues']
        feedback.append(f"Function '{function_name}':")
        for issue in issues:
            if 'Excessive nesting detected' in issue:
                feedback.append(f"  - Reduce the nesting levels in '{function_name}' to improve readability.")
            if 'Missing docstring' in issue:
                feedback.append(f"  - Add a docstring to '{function_name}' to improve documentation.")
            if 'Cyclomatic complexity' in issue:
                feedback.append(f"  - Simplify the logic in '{function_name}' to reduce cyclomatic complexity.")
            if 'Does not follow snake_case naming convention' in issue:
                feedback.append(f"  - Rename '{function_name}' to follow snake_case naming convention.")
            if 'No inline comments' in issue:
                feedback.append(f"  - Add inline comments to '{function_name}' to explain complex logic.")
            if 'Contains dangerous functions' in issue:
                feedback.append(f"  - Avoid using dangerous functions like eval or exec in '{function_name}'.")
        feedback.append(f"  - Control Flow Statements: {', '.join(finding['control_flow'])}")
        feedback.append(f"  - Data Dependencies: {', '.join(finding['data_dependencies'])}")
    return feedback

def print_findings(findings: List[Dict[str, Any]]) -> None:
    """
    Print the findings in a readable format.

    Parameters:
    findings (List[Dict[str, Any]]): The list of findings for each function.
    """
    for finding in findings:
        logging.info(f"Function: {finding['function_name']}")
        logging.info(f"Start Point: {finding['start_point']}")
        logging.info("Issues:")
        for issue in finding['issues']:
            logging.info(f"- {issue}")
        logging.info("Control Flow Statements:")
        logging.info(f"- {', '.join(finding['control_flow'])}")
        logging.info("Data Dependencies:")
        logging.info(f"- {', '.join(finding['data_dependencies'])}")
        logging.info("AST:")
        logging.info(json.dumps(finding['ast'], indent=2))
        logging.info("")

# Example usage
code = """
def foo():
    if bar:
        eval("baz()")

def bar():
    pass

def nested_example():
    if cond1:
        if cond2:
            if cond3:
                pass
"""

findings = analyze_code(code)
print_findings(findings)
quality = evaluate_code_quality(findings)
logging.info(f"Code Quality: {quality}")

# Provide feedback
feedback = provide_feedback(findings)
for line in feedback:
    logging.info(line)
