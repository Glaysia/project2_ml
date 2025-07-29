class ProblemVariables:

    def __init__(self, vars_dict=None):
        # default : empty dict
        self.vars = vars_dict if vars_dict is not None else {}

    def get_names(self):
        return [name for name in self.vars.keys()]

    def get_first_values(self):
        return [values[0] for values in self.vars.values()]

    def get_second_values(self):
        return [values[1] for values in self.vars.values()]
    
    def get_scale_values(self):
        return [values[2] for values in self.vars.values()]
    
    def get_unit_scale_values(self):
        return [values[3] for values in self.vars.values()]
    
    def get_num_of_variables(self):
        return len(self.vars)

    def add_variable(self, name, values):
        self.vars[name] = values
    
    def update_variable(self, name, values):
        if name in self.vars:
            self.vars[name] = values
    
    def delete_variable(self, name):
        if name in self.vars:
            del self.vars[name]



