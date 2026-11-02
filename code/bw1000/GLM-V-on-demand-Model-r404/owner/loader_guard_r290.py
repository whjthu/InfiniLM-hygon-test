"""Guard only weight I/O/commit call boundaries; preserve original loader math."""
import ast

def guard_loader(text):
 tree=ast.parse(text);counts={'safe_open':0,'modern_glm_model_load_batch':0,'modern_glm_model_finish':0}
 class Guard(ast.NodeTransformer):
  def visit_Call(self,node):
   self.generic_visit(node);name=node.func.id if isinstance(node.func,ast.Name)else node.func.attr if isinstance(node.func,ast.Attribute)else None
   if name not in counts:return node
   counts[name]+=1
   return ast.copy_location(ast.Subscript(value=ast.Tuple(elts=[ast.Call(func=ast.Name(id='__weight_deadline_check_r290__',ctx=ast.Load()),args=[ast.Constant(value=name)],keywords=[]),node],ctx=ast.Load()),slice=ast.Constant(value=1),ctx=ast.Load()),node)
 Guard().visit(tree);assert counts=={'safe_open':1,'modern_glm_model_load_batch':1,'modern_glm_model_finish':1},counts
 return ast.fix_missing_locations(tree),counts
