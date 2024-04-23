from data_modules.sst import SSTDataModule
from data_modules.imdb import ImdbDataModule
from data_modules.yahoo import YahooDataModule
from data_modules.trec import TRECDataModule
from data_modules.agnews import AgNewsDataModule


available_data_modules = {
    "sst": SSTDataModule,
    "imdb": ImdbDataModule,
    "yahoo": YahooDataModule,
    "trec": TRECDataModule,
    "agnews": AgNewsDataModule
}