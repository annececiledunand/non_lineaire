import click
import os


import BE1, BE2, BE3



@click.group()
def cli():
    pass


@cli.command()
def test(name='test'):
    print("Bonjour ! Tout s'est bien installé.")


@cli.command(name='run-be1')
@click.argument('Exercice')
def execute_be1(exercice: str):
    print(f'Launching Exercice {exercice} in BE 3')
    if exercice == '1':
        BE1.Exercice1()
    elif exercice == '2':
        BE1.Exercice2()
    elif exercice == '3':
        BE1.Exercice3()
    else:
        print(f"Exercice nb {exercice} doesn\'t exist")

@cli.command(name='run-be2')
@click.argument('Exercice')
def execute_be1(exercice: str):
    print(f'Launching Exercice {exercice} in BE 2')
    if exercice == '1':
        BE2.Exercice1()
    elif exercice == '2':
        BE2.Exercice2()
    elif exercice == '3':
        BE2.Exercice3()
    else:
        print(f"Exercice nb {exercice} doesn\'t exist")


@cli.command(name='run-be3')
@click.argument('Exercice')
def execute_be1(exercice: str):
    print(f'Launching Exercice {exercice} in BE 3')
    if exercice == '1':
        BE3.Exercice1()
    else:
        print(f"Exercice nb {exercice} doesn\'t exist")



if __name__ == '__main__':
    cli()
