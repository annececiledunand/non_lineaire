import click
import os

from BEs import BE1, BE2, BE3


@click.group()
def cli():
    print('Les scripts sont un peu long à tourner.')
    print('Pensez bien à fermer les fenetres pour continuer à faire tourner le script.\n')


@cli.command()
def test(name='test'):
    print("Bonjour ! Tout s'est bien importé.")


@cli.command(name='run-be1')
@click.argument('Exercice')
def execute_be1(exercice: str):
    print(f'Lancement de l\'exercice {exercice} in BE 1\n')
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
    print(f'Lancement de l\'exercice {exercice} in BE 2\n')
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
    print(f'Lancement de l\'exercice {exercice} in BE 3\n')
    if exercice == '1':
        BE3.Exercice1()
    else:
        print(f"Exercice nb {exercice} doesn\'t exist")



if __name__ == '__main__':
    cli()
