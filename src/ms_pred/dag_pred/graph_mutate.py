import random
from functools import partial
from typing import Optional, List, Callable

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, RWMol, rdmolops

rdBase.DisableLog("rdApp.error")

def run_rxn(mols: List[Chem.Mol], smarts: str) -> List[Chem.Mol]:
    """ run_rxn helper function"""
    rxn = AllChem.ReactionFromSmarts(smarts)
    outs = rxn.RunReactants(mols)
    return [j for i in outs for j in i]

def downgrade_bond(mol: Chem.Mol) -> Optional[Chem.Mol]: 
    """Choose a bond of order >1 & not in ring and downgrade it.
        Return:
        
        New molecule with downgraded bond	

    """
    Chem.Kekulize(mol, clearAromaticFlags=True)
    m_edit = RWMol(mol)
    match_struct = Chem.MolFromSmarts("[*]!-;!@[*]")
    matching_inds = m_edit.GetSubstructMatches(match_struct)
    if len(matching_inds) == 0: 
        return None
    
    # [Zero, Single, Double, Triple]
    bond_prob_vec = np.array([0, 0, 1, 0.5])

    # Smarter sampling
    bonds = [m_edit.GetBondBetweenAtoms(*i) for i in matching_inds]
    bond_types = np.array([x.GetBondType() for x in bonds])
    bond_probs = bond_prob_vec[bond_types]
    bond_probs = bond_probs / bond_probs.sum()

    # Sample
    ind = np.random.choice(len(bonds), p=bond_probs)
    bis = matching_inds[ind]    

    # Find bond sampled and get new bond type
    b = m_edit.GetBondBetweenAtoms(bis[0], bis[1])
    b_type = b.GetBondType()
    b_type_new = b_type - 1

    # Remove bond
    m_edit.RemoveBond(*bis)
    
    m_edit.AddBond(*bis, Chem.BondType(b_type_new))

    # Update 
    # Add two new hydrogens
    a1 = Chem.Atom("H")
    a2 = Chem.Atom("H")
    
    ind_0 = m_edit.AddAtom(a1)
    ind_1 = m_edit.AddAtom(a2)
    
    m_edit.AddBond(bis[0], ind_0, Chem.BondType(1))
    m_edit.AddBond(bis[1], ind_1, Chem.BondType(1))
    mol = Chem.Mol(m_edit)

    new_mol = Chem.RemoveHs(m_edit)
    return new_mol

def upgrade_bond(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """upgrade_bond.

	Choose a bond of order <3, not in ring, and with >=1H to upgrade

	Return:
		New molecule with upgraded bond	

	"""
    Chem.Kekulize(mol, clearAromaticFlags=True)

    m_edit = RWMol(mol)
    query = "[*;!H0]~;!#;!@[*;!H0]"

    # bonded, not triple, and not aromatic
    match_struct = Chem.MolFromSmarts(query)
    matching_inds = m_edit.GetSubstructMatches(match_struct)
    if len(matching_inds) == 0:
        return None
#     matching_inds.extend(m_edit.GetSubstructMatches(match_struct_2))

    # [Zero, Single, Double, Triple]
    # Potentially look into ring vs. non ring bonds if not aromatic...
    # Don't add 
    bond_prob_vec = np.array([0, 1, 0.05, 0.00])

    # Smarter sampling
    bonds = [m_edit.GetBondBetweenAtoms(*i) for i in matching_inds]
    bond_types = np.array([x.GetBondType() for x in bonds])
    bond_probs = bond_prob_vec[bond_types]
    bond_probs = bond_probs / bond_probs.sum()

    # Sample
    ind = np.random.choice(len(bonds), p=bond_probs)
    bis = matching_inds[ind]    

    # Find bond sampled and get new bond type
    b = m_edit.GetBondBetweenAtoms(bis[0], bis[1])
    b_type = b.GetBondType()
    b_type_new = b_type + 1
    
    # Remove H's in preparation
    m_edit = RWMol(Chem.AddHs(m_edit))
    for idx in bis:
        atom = m_edit.GetAtomWithIdx(idx)
        
        # Remove a neighboring hydrogen
        for a in atom.GetNeighbors():
            if a.GetAtomicNum() == 1:
                m_edit.RemoveAtom(a.GetIdx())
                break

    # Remove bond
    m_edit.RemoveBond(*bis)
    m_edit.AddBond(*bis, Chem.BondType(b_type_new))
    return Chem.RemoveHs(m_edit)

def break_ring(mol: Chem.Mol):
    """ break_ring."""
    # Single only!
    Chem.Kekulize(mol, clearAromaticFlags=True)
    cyc_smarts = "[*:1]-;@[*:2]>>([*:1].[*:2])"
    out_l = list(run_rxn([mol], cyc_smarts))
    if len(out_l) == 0:
        return None
    
    new_mol = (random.choice(out_l))

    if not mol_ok(new_mol): # will check + sanitize mol
        return None
    else:
        return new_mol

def make_ring(mol):
    """make_ring."""
    choices = [
        "[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1",
        "[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1",
    ]
    
    p = [0.05, 0.05, 0.45, 0.45]

    smarts = np.random.choice(choices, p=p)

    Chem.Kekulize(mol, clearAromaticFlags=True)
    
    out_l = list(run_rxn([mol], smarts))
    if len(out_l) == 0:
        return None
    
    new_mol = (random.choice(out_l))
    if not mol_ok(new_mol): # will check + sanitize mol
        return None
    else:
        return new_mol

def cut_and_paste(mol: Chem.Mol) -> Optional[Chem.Mol]: 
    """cut_and_paste.

    Take section from one part of molecule and rejoin it to another
    """ 

    Chem.Kekulize(mol, clearAromaticFlags=True)
    m_edit = RWMol(Chem.AddHs(mol))
    
    # Find a single bond
    match_struct = Chem.MolFromSmarts("[*;!#1]-;!@[*;!#1]")
    matching_inds = m_edit.GetSubstructMatches(match_struct)
    if len(matching_inds) ==  0: 
        return None
    
    # Find all possible acceptors
    h_struct = Chem.MolFromSmarts("[!H0]")
    h_match = m_edit.GetSubstructMatches(h_struct)
    h_match = set([i[0] for i in h_match])
    if len(h_match) ==  0: 
        return None
    
    # Choose a bond to break
    bond = list(random.choice(matching_inds))
    random.shuffle(bond)
    orig_src, leaving = bond
    
    # Delete the first bond and pad Hs
    # Remove bond
    m_edit.RemoveBond(orig_src, leaving)
    
    # Create bond from orig_src to new hydrogen
    a1 = Chem.Atom(1)
    new_h = m_edit.AddAtom(a1)
    m_edit.AddBond(orig_src, new_h, Chem.BondType(1))

    # Filter down the list of potential acceptors
    # Find all connected components starting at oric_src to make sure we
    # don't make two fragments
    mol_frags = rdmolops.GetMolFrags(m_edit)
    unreachable = [j for i in mol_frags for j in i if orig_src not in i]
    invalid_els = set()
    invalid_els.update(
        [i.GetIdx() for i in m_edit.GetAtomWithIdx(leaving).GetNeighbors()]
    )
    invalid_els.update([leaving, orig_src])
    invalid_els.update(unreachable)
    
    # Remove leaving and all its neighbors from attach trg options
    h_match.difference_update(invalid_els)    
    
    # Choose trg
    if len(h_match) == 0: 
        return None
    trg = random.choice(list(h_match))
    
    # Construct a bond from leaving to trg
    m_edit.AddBond(leaving, trg, Chem.BondType(1))
    
    # Remove H on trg
    m_edit.GetAtomWithIdx(trg).GetNeighbors()
    for a in m_edit.GetAtomWithIdx(trg).GetNeighbors():
        if a.GetAtomicNum() == 1:
            m_edit.RemoveAtom(a.GetIdx())
            break
        
    new_mol = Chem.RemoveHs(m_edit)
    return new_mol

def mol_ok(mol):
    """ mol_ok. """
    try:
        Chem.SanitizeMol(mol)
        return True
    except ValueError:
        return False

def ring_ok(mol):
    """ ring_ok. 

    Change to allow more flexibility 

    """

    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))

    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(j) for j in cycle_list])
    macro_cycle = max_cycle_length > 6

    double_bond_in_small_ring = mol.HasSubstructMatch(
        Chem.MolFromSmarts("[r3,r4]=[r3,r4]")
    )


    return not ring_allene and not macro_cycle and not double_bond_in_small_ring

def apply_down_up(mol: Chem.Mol, fn_down: Callable, fn_up: Callable):
    """ apply_down_up.

    Partial function that jointly applies a bond downgrade and upgrade

    Args:
        mol:
        fn_down
        fn_up 
    """
    down_out = fn_down(mol)
    if down_out is None: 
        return None

    up_out = fn_up(down_out)
    if up_out is None: 
        return None
    
    return up_out

def mutate(mol, mutation_rate):
    """ mutate."""

    # Create pairs of functions with down and up
    if random.random() > mutation_rate:
        return mol

    transforms = [
		partial(apply_down_up, fn_down = downgrade_bond, fn_up = upgrade_bond),
		partial(apply_down_up, fn_down = downgrade_bond, fn_up = make_ring),
		partial(apply_down_up, fn_down = break_ring, fn_up = upgrade_bond),
		partial(apply_down_up, fn_down = break_ring, fn_up = make_ring),
		cut_and_paste,
    ]
    transforms_p = [0.03,0.14,0.03,0.2,0.6]

    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        return mol

    for i in range(10):
        transform = np.random.choice(transforms, p=transforms_p)
        new_mol = transform(mol)
        if new_mol is None:
            continue
        elif mol_ok(new_mol) and ring_ok(new_mol): 
            return new_mol
        else: 
            continue
    return None
