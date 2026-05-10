from src.agents.inspector import run_inspector
from src.models import ColumnType

def afficher_structure(structure):
    print("\n" + "="*60)
    print(f"  RAPPORT D'INSPECTION : {structure.file_name}")
    print("="*60)
    print(f"  Total cellules : {structure.total_cells:,}")
    print(f"  Feuilles totales : {len(structure.sheets)}")
    print(f"  Feuilles pertinentes : {len(structure.relevant_sheets)}")

    for sheet in structure.sheets:
        pertinence = "✅" if sheet.is_relevant else "❌"
        print(f"\n{'─'*60}")
        print(f"  {pertinence} Feuille : '{sheet.name}'")
        print(f"     Dimensions : {sheet.n_rows} lignes × {sheet.n_cols} colonnes")
        print(f"     En-tête à la ligne : {sheet.header_row_index}")

        if sheet.structure_notes:
            for note in sheet.structure_notes:
                print(f"     ⚠️  {note}")

        if sheet.is_relevant:
            print(f"     Colonnes :")
            for col in sheet.columns:
                ignore_tag = " [IGNORÉE]" if col.should_ignore else ""
                print(f"       • {col.name:<25} {col.column_type.value:<12}"
                      f" null:{col.null_ratio:.0%}{ignore_tag}")
                if col.mean is not None:
                    print(f"         → moy={col.mean}, "
                          f"std={col.std}, "
                          f"min={col.min_val}, "
                          f"max={col.max_val}")


if __name__ == "__main__":
    structure = run_inspector("data/samples/mesures_centrale.xlsx")
    afficher_structure(structure)