import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('package_v3',Path(__file__).resolve().parents[1]/'scripts/package_public_capacity_result_v3.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class PackageCases(unittest.TestCase):
    def fixture(self):
        result={'schema':'public-dictionary-full-development-v3-result','metadata':{'unknown':None,'zero':0,'unicode':'Ёмкость'},'extra_future_field':[1,{'x':True}]}
        result.update({field:[{'ordinal':i,'capacity':None if i%2 else 0,'slots':[{'surface':'word','width':1}]} for i in range(n)] for field,n in p.TABLES.items()})
        return result

    def test_every_ordered_row_and_extra_metadata_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'package';result=self.fixture();manifest=p.pack(result,root)
            self.assertEqual(p.reconstruct(root,manifest),result)

    def test_tampered_part_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'package';manifest=p.pack(self.fixture(),root)
            target=root/manifest['tables']['lengths']['parts'][0]['path'];target.write_bytes(target.read_bytes()+b'{}\n')
            with self.assertRaises(ValueError):p.reconstruct(root,manifest)

    def test_missing_null_rows_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=self.fixture();result['dictionary_rows'].pop()
            with self.assertRaises(ValueError):p.pack(result,Path(tmp)/'package')


if __name__=='__main__':unittest.main()
