import pyeliza

class Eliza:
    aliases = 'eliza'
    description = 'Virtual therapist'
    _therapist = pyeliza.eliza()

    def execute(self, expression, context):
        '''
        >>> from mock import Mock

        >>> eliza = Eliza()

        >>> context = Mock()

        >>> context.claim_for = Mock()
        >>> def set_owner(owner):
        ...     context.owner = owner
        >>> context.claim_for.side_effect = set_owner

        >>> context.release = Mock()
        >>> def release():
        ...     context.owner = None
        >>> context.release.side_effect = release

        Run Eliza
        >>> eliza.execute('eliza', context)
        'Hello. How are you feeling today?'
        >>> isinstance(context.owner, Eliza)
        True

        Respond
        >>> eliza.execute('Fine', context)
        '...'

        Quit
        >>> eliza.execute('    quit   ', context)
        '...'
        >>> context.owner is None
        True
        '''
        expression = expression.strip()

        if expression == 'quit':
            context.release()
        elif context != self:
            context.claim_for(self)
            
            return 'Hello. How are you feeling today?'

        return self._therapist.respond(expression)

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
