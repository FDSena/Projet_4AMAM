class Option:

    def __init__(self, strike, maturity, option_type):
        if option_type not in ('call', 'put'):
            raise ValueError(f"option_type doit être 'call' ou 'put', reçu : '{option_type}'")
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type

    def payoff(self, S):
        if self.option_type == 'call':
            return max(S - self.strike, 0)
        else:
            return max(self.strike - S, 0)

    def __repr__(self):
        return f"Option(type='{self.option_type}', strike={self.strike}, maturity={self.maturity})"