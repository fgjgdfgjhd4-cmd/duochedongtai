class Genome(object):
    def __init__(self, initial_solution, initial_fitness):
        super(Genome, self).__init__()

        self.solution = initial_solution
        self.fitness = initial_fitness

    def __repr__(self):
        return f'<Genome g:{self.solution} f:{self.fitness} />'
