import requests
from urllib.parse import urljoin
import time
import pprint
from typing import List, Tuple, Dict, Union
from pathlib import Path
from abc import ABC, abstractmethod
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from rdkit import Chem
from tqdm import tqdm
import warnings
import pandas as pd

from ms_pred import common

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

pp = pprint.PrettyPrinter(indent=4)


class Coster(ABC):
    """
    Determines whether a molecule represented as a SMILES is
    buyable. If it is, determines its cost
    TODO: implement LookupCoster so user can input csv of inventory with cost=0
    or maybe allow ChemSpaceCoster to be updated with inventory list and
    change those buyables to cost = 0
    """

    def __init__(self):
        self.status_log = None

    @abstractmethod
    def __call__(smis: List[str]) -> Tuple[Dict, List]:
        """ For a list of smiles, returns a dictionary that maps the smiles to their cost
         and a list of the smiles that are buyable. """

    @abstractmethod
    def get_buyable_and_cost(smiles: str) -> Tuple[bool, float]:
        """ For a single smiles strings, outputs if the smiles is buyable and what its cost is"""

    def build_status_log(self, pos=1):
        self.status_log = tqdm(total=0, position=pos, bar_format='{desc}')


class NaiveCoster(Coster):
    """ Determines if a molecule is buyable based on the number of certain elements.
    Sets all costs to 1
    """

    def __init__(self) -> None:
        warnings.warn('Using a naive method to assign buyability and cost with no guarantee of accuracy.')
        return

    def buyable(self, smiles: str) -> bool:

        nc = smiles.count("c") + smiles.count("C")
        nn = smiles.count("N") + smiles.count("n")
        no = smiles.count("O") + smiles.count("o")

        if nc <= 12 and nn <= 3 and no <= 5:
            return True
        else:
            return False

    def get_buyable_and_cost(self, smiles: str) -> Tuple[bool, float]:

        buyable = self.buyable(smiles)

        if buyable:
            return True, 1
        else:
            return False, None

    def __call__(self, smis: List[str]) -> Tuple[Dict, set]:
        costs = {}
        buyables = set()

        for smiles in smis:
            buyable = self.buyable(smiles)
            if buyable:
                costs[smiles] = 1
                buyables.add(smiles)
            else:
                costs[smiles] = None
                buyables.add(smiles)

        return costs, buyables


class ChemSpaceCoster(Coster):
    """ Determines if a molecule is buyable and what its cost is (if buyable)
    Adapted from PyChemSearch (Marco Stenta, Marta Pasquini @ Syngenta)"""

    def __init__(self,
                 api_key,
                 base_url='https://api.chem-space.com/',
                 api_version='v3') -> None:

        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version

        self.token_expiration_time = 3600
        self.get_token()

        self.search_ref_time = time.time()
        self.searches_in_time_frame = 0
        self.max_search_in_time_frame = 35
        self.time_frame = 60

        self.categories_map = {
            # 'CSMS': 'Make-On-Demand Screening Compounds',
            # 'CSMB': 'Make-On-Demand Building Blocks',
            # 'CSCS': 'Custom Request', # only care about in stock
            'CSSB': 'In-Stock Building Blocks',
            'CSSS': 'In-stock Screening Compounds'
        }

        self.filters = {
            'purity': 90,
            'shipsWithin': 60,
            'min_packMg': 5,
            'max_packMg': 10000,
        }  # TODO: make this user-set parameters

        super().__init__()

        return

    def check_time_and_token(self):
        if (time.time() - self.search_ref_time) > 59:
            self.search_ref_time = time.time()
            self.searches_in_time_frame = 0
            self.status_log.set_description_str('')
        elif self.searches_in_time_frame >= self.max_search_in_time_frame:  # out of searches in this minute
            self.status_log.set_description_str('Waiting to avoid hitting max requests')
            time.sleep(65 - (time.time() - self.search_ref_time))  # sleep until you can search more
            self.searches_in_time_frame = 0
            self.search_ref_time = time.time()
            self.status_log.set_description_str('')

        delta_time = int(time.time() - self.token_ref_time)

        if delta_time > self.token_expiration_time:
            self.get_token()

        return

    def get_token(self):
        headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer {}'.format(self.api_key),
        }

        url = urljoin(urljoin(self.base_url, self.api_version), 'auth/token')
        r = requests.get(url, headers=headers, verify=False)
        response = r.json()

        if 'access_token' in response:
            token = response['access_token']
        else:
            print(response)
            token = None

        self.token_ref_time = time.time()  # set the reference time: the token can be used within
        self.token_requests = 0
        self.token = token

    def single_search(self, smiles, max_results_count=100,
                      categories='CSSB, CSSS'):

        # if the time difference between now and the reference time (set at token creation)
        # is larger than expiration time, a new token is requested

        self.check_time_and_token()
        self.token_requests += 1
        self.searches_in_time_frame += 1

        headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer {}'.format(self.token),
        }

        params = (
            ('count', max_results_count),
            ('categories', categories),
        )

        files = {
            'SMILES': (None, smiles),
            # 'InChIKey': (None, common.inchikey_from_smiles(smiles))
        }

        url = urljoin(self.base_url, '{}/search/{}'.format(self.api_version, 'exact'))
        # url = urljoin(self.base_url, '{}/search/{}'.format(self.api_version, 'text'))

        response = None
        for i in range(5): # max retries
            try:
                response = requests.post(url, headers=headers, params=params, files=files, verify=False)
            except Exception as err:
                print(f'Get Error, retrying {i}:', err)
            if response is not None:
                break

        processed_response = self.process_response(response)

        return processed_response

    def cost_from_response(self, response, must_have_usd=False):
        if response['status_code'] == 429 or response['status_code'] == 401:
            self.get_token()
            return None

        offers = [offer for item in response['content']['items'] for offer in item['offers']]

        if len(offers) == 0:
            return None

        filtered_offers = []

        for offer in offers:
            if self.offer_filter(offer):
                for price in offer['prices']:
                    if must_have_usd:
                        if price['priceUsd'] is not None:
                            filtered_offers.append(dict(shipsWithin=offer['shipsWithin'], **price))
                    else:
                        filtered_offers.append(dict(shipsWithin=offer['shipsWithin'], **price))

        prices = [(offer['priceUsd'], offer['packMg'], offer['shipsWithin']) for offer in filtered_offers if self.price_filter(offer)]  # USD, packMg, shipsWithin

        if len(prices) == 0:
            return None

        def sort_func(inp):
            if inp[0] is not None:
                return inp[0]
            elif inp[1] is not None:
                return inp[1] + 100000 # make sure always ranked lower
            else:
                raise ValueError
        return sorted(prices, key=sort_func)[0]


    def offer_filter(self, offer):
        if 'shipsWithin' in self.filters and int(offer['shipsWithin']) > self.filters['shipsWithin']:
            return False
        if 'purity' in self.filters and offer['purity'] < self.filters['purity']:
            return False
        return True

    def price_filter(self, price):
        if 'min_packMg' in self.filters and price['packMg'] < self.filters['min_packMg']:
            return False
        if 'max_packMg' in self.filters and price['packMg'] > self.filters['max_packMg']:
            return False
        # if 'max_rawCost' in self.filters and price['priceUsd'] > self.filters['max_rawCost']:
        #     return False
        return True

    def process_response(self, response):
        if response.status_code == 200:
            content = response.json()
        else:
            content = None
            print('msg from process_response():', 'status_code', response.status_code, 'reason', response.reason)

        return {'content': content, 'status_code': response.status_code, 'reason': response.reason}

    def get_buyable_and_cost(self, smiles: str, in_stock_only=True) -> Tuple[bool, float]:
        if in_stock_only:
            categories = 'CSSB, CSSS'
        else:
            categories = 'CSSB, CSSS, CSCS'
        response = self.single_search(smiles, categories=categories)
        cost = self.cost_from_response(response)
        if cost is None:
            return False, None
        else:
            return True, cost

    def __call__(self, smis: List[str]):

        costs = {}
        buyables = set()

        for smiles in smis:
            response = self.single_search(smiles)
            cost = self.cost_from_response(response)
            costs[smiles] = cost
            if cost is not None:
                buyables.add(smiles)

        return costs, buyables


class LookupCoster(Coster):
    """
    Determines if a molecule is buyable and what its cost is
    by looking up the smiles in a csv file. By default, assumes the zeroth column
    contains SMILES strings and the first column contains costs.
    """

    def __init__(self,
                 lookup_file: Union[str, Path],
                 smis_col: int = 0,
                 cost_col: int = 1,
                 canonicalize: bool = True
                 ):
        try:
            self.data = pd.read_csv(lookup_file)
        except:
            self.data = pd.read_csv(lookup_file, lineterminator='\n')

        self.buyables = self.data.iloc[:, smis_col]
        self.failures = []

        if canonicalize:
            self.buyables = self.canon_library(self.buyables)

        self.costs = self.data.iloc[:, cost_col]
        self.cost_map = {
            bb: cost for bb, cost
            in zip(self.buyables, self.costs)
        }

        print(f'Inventory loaded with {len(self.cost_map)} compounds')
        print(f'Remaining {len(self.costs) - len(self.cost_map)} were either duplicates or unable to be canonicalized')

    def canon_library(self, smi_list: list) -> list:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        fn = lambda x: self.canonicalize(x)
        canon_smis = common.chunked_parallel(smi_list, function=fn, chunks=8, desc='Canonicalizing buyables')
        # canon_smis = [self.canonicalize(smi) for smi in tqdm(smi_list, desc='Canonicalizing buyables...')]
        return canon_smis

    def canonicalize(self, smi: str) -> str:
        try:
            smicanon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        except:
            smicanon = None
            # self.failures.append(smi)
        return smicanon

    def get_buyable_and_cost(self, smiles: str) -> Tuple[bool, float]:
        if smiles in self.cost_map:
            return True, self.cost_map[smiles]
        canonsmi = self.canonicalize(smiles)
        if canonsmi in self.cost_map:
            return True, self.cost_map[canonsmi]
        return False, None

    def __call__(self, smis: List[str]):

        costs = {}
        buyables = set()

        for smiles in smis:
            _, cost = self.get_buyable_and_cost(self, smiles)
            costs[smiles] = cost
            if cost is not None:
                buyables.add(smiles)

        return costs, buyables


class QuantityLookupCoster(LookupCoster, Coster):
    """
    Similar to a LookupCoster, this Coster references a CSV file with a list of SMILES and their
    costs. A QuantityAwareERSelector will use
    a QuantityLookupCoster to get cost as a function of quantity.

    In this Coster, though, the CSV file should contain multiple columns corresponding to
    different quantities. NaN or None entries are acceptable, but not as column names.
    The zeroth row is assumed to be a header row. The zeroth column is assumed to contain
    SMILES strings of buyable compounds. The header for each subsequent column should be
    interpretable as a float and is assumed to be in grams. For example, this class
    currently does not support headers such as '50mg' or '0.05g'. Instead use '0.05' as
    the header for the column which contains costs for 50mg.

    To return a single cost, this Coster returns the $/g for the smallest quantity.
    """

    def __init__(self,
                 lookup_file: Union[str, Path],
                 smis_col: int = 0,
                 canonicalize: bool = True
                 ):

        try:
            self.data = pd.read_csv(lookup_file)
        except:
            self.data = pd.read_csv(lookup_file, lineterminator='\n')

        self.buyables_precanon = self.data.iloc[:, smis_col]
        self.failures = []

        if canonicalize:
            self.buyables = self.canon_library(self.buyables)
        else:
            self.buyables = self.buyables_precanon

        self.inventory = {
            canon_smi: dict(self.data.loc[smi])
            for canon_smi, smi in zip(self.buyables, self.buyables_precanon)
        }

        print(f'Inventory loaded with {len(self.inventory)} compounds')
        print(
            f'Remaining {len(self.buyables_precanon) - len(self.inventory)} were either duplicates or unable to be canonicalized')

    def get_buyable_and_cost(self, smiles):
        costs = self.get_cost_function(smiles)
        if costs is None or len(costs) == 0:
            return False, None
        return True, costs[0][1] / costs[0][0]

    def get_cost_function(self, smiles):
        if smiles in self.inventory:
            costs = [(float(amt), cost) for amt, cost in self.inventory[smiles].items() if str(cost) != 'nan']
            costs = sorted(costs)
            return costs
        return None