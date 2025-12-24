import os

from dotenv import load_dotenv
load_dotenv(".env", override=True)
lang_code: str = os.getenv("LOCALIZATION_LANGUAGE") or "en-US"

class str:
    app_title = {
            "en-US": f'Geo Parametrization',
            "pt-BR": f'Geo Parametrização',
        }[lang_code]
    select_action = {
            "en-US": f'Select an action',
            "pt-BR": f'Selecione uma ação',
        }[lang_code]
    welcome = {
            "en-US": f'Welcome to the Geological Data Collection App',
            "pt-BR": f'Bem-vindo ao App de Coleta de Dados Geológicos',
        }[lang_code]
    params_c_k_segment_porosity = {
            "en-US": f'Parameterization for image segmentation (C&K from CMYK)',
            "pt-BR": f'Parametrização para segmentação de imagens (C&K do CMYK)',
        }[lang_code]
    exit = {
            "en-US": f'Exit',
            "pt-BR": f'Sair',
        }[lang_code]
    enter = {
            "en-US": f'Enter',
            "pt-BR": f'Entrar',
        }[lang_code]
    new_user = {
            "en-US": f'New User',
            "pt-BR": f'Novo Usuário',
        }[lang_code]
    statistics = {
            "en-US": f'Statistics',
            "pt-BR": f'Estatísticas',
        }[lang_code]
    best_color_params_plot_title = {
            "en-US": f'Channel tuple selected by user',
            "pt-BR": f'Tuplas de Canais Selecionadas pelos Usuários',
        }[lang_code]
    k_channel_max = {
            "en-US": f'K Channel (maximum)',
            "pt-BR": f'Canal K (máximo)',
        }[lang_code]
    c_channel_min = {
            "en-US": f'C Channel (minimum)',
            "pt-BR": f'Canal C (mínimo)',
        }[lang_code]
    clicked_points = {
            "en-US": f'Clicked Points',
            "pt-BR": f'Pontos Clicados',
        }[lang_code]
    experience = {
            "en-US": f'Experience',
            "pt-BR": f'Experiência',
        }[lang_code]
    experience_1_no_experience = {
            "en-US": f'no experience',
            "pt-BR": f'não possuo experiência',
        }[lang_code]
    experience_2_studying = {
            "en-US": f'I am studying the subject',
            "pt-BR": f'estou estudando o assunto',
        }[lang_code]
    experience_3_experienced = {
            "en-US": f'done this a few times',
            "pt-BR": f'já fiz algumas vezes',
        }[lang_code]
    experience_4_related_area = {
            "en-US": f'I work in a related area',
            "pt-BR": f'trabalho em área relacionada',
        }[lang_code]
    experience_5_work_with_this = {
            "en-US": f'I work with this all the time',
            "pt-BR": f'trabalho nisso o tempo todo',
        }[lang_code]
    experience_levels = [
        experience_1_no_experience,
        experience_2_studying,
        experience_3_experienced,
        experience_4_related_area,
        experience_5_work_with_this,
    ]
    experience_level = {
            "en-US": lambda exp: f'Experience Level {exp}',
            "pt-BR": lambda exp: f'Nível de Experiência {exp}',
        }[lang_code]
    porosity = {
            "en-US": f'Porosity',
            "pt-BR": f'Porosidade',
        }[lang_code]
    cumulative_probability = {
            "en-US": f'Cumulative Probability',
            "pt-BR": f'Probabilidade Acumulada',
        }[lang_code]
    cdf_porosity_by_experience = {
            "en-US": f'CDF of Porosity by Experience Level',
            "pt-BR": f'CDF de Porosidade por Nível de Experiência',
        }[lang_code]
    pore_count = {
            "en-US": f'Pore Count',
            "pt-BR": f'Contagem de Poros',
        }[lang_code]
    cdf_pore_count_by_experience = {
            "en-US": f'CDF of Pore Count by Experience Level',
            "pt-BR": f'CDF de Contagem de Poros por Nível de Experiência',
        }[lang_code]
    userinfo_email_required = {
            "en-US": f'Please fill in the email field.',
            "pt-BR": f'Por favor, preencha o campo de e-mail.',
        }[lang_code]
    select_experience_level_range = {
            "en-US": f'Select Experience Level Range',
            "pt-BR": f'Selecione o Intervalo de Nível de Experiência',
        }[lang_code]
    select_min_pore_size_range = {
            "en-US": f'Select pore size range',
            "pt-BR": f'Selecione o intervalo de tamanho dos poros',
        }[lang_code]
    end_report = {
            "en-US": f'END OF REPORT',
            "pt-BR": f'FIM DO RELATÓRIO',
        }[lang_code]
    fill_all_fields = {
            "en-US": f'Please fill in all fields.',
            "pt-BR": f'Por favor, preencha todos os campos.',
        }[lang_code]
    data_saved = {
            "en-US": f'Data saved successfully!',
            "pt-BR": f'Dados salvos com sucesso!',
        }[lang_code]
    your_login = {
            "en-US": lambda name: f'You are logged in as {name}',
            "pt-BR": lambda name: f'Você está logado como {name}',
        }[lang_code]
    use_canceled_or_tests = {
            "en-US": f'Use canceled or test sessions',
            "pt-BR": f'Usar sessões canceladas ou de teste',
        }[lang_code]
    class texts:
        welcome_paragraph = {
                "en-US": f"""
                Welcome to the Geological Data Collection App.
                This app allows you to collect parameters for
                understanding geological data, such as petrographic
                slim-sections.
                """,
                "pt-BR": f"""
                Bem-vindo ao App de Coleta de Dados Geológicos.
                Este aplicativo permite o recolhimento de parâmetros
                para entendimento de dados geológicos, como lâminas
                petrográficas.
                """
        }[lang_code]
        more_instructions = {
                "en-US": f"""
                By entering the application, you will create a session
                and will be presented to
                more options to collect data and parameters about
                geological features.
                """,
                "pt-BR": f"""
                Ao entrar no aplicativo, você criará uma sessão
                e será apresentado a mais opções para coletar dados
                e parâmetros sobre características geológicas.
                """
        }[lang_code]
    select_experience_levels = {
            "en-US": f'Select Experience Levels to Show',
            "pt-BR": f'Selecione os Níveis de Experiência a Mostrar',
        }[lang_code]
    show_means = {
            "en-US": f'Show Means',
            "pt-BR": f'Mostrar Médias',
        }[lang_code]
    show_dispersions = {
            "en-US": f'Show Dispersions',
            "pt-BR": f'Mostrar Dispersões',
        }[lang_code]
    show_aggregations = {
            "en-US": f'Show Aggregation',
            "pt-BR": f'Mostrar Agregação',
        }[lang_code]
    plot_options = {
            "en-US": f'Plot Options',
            "pt-BR": f'Opções de Gráfico',
        }[lang_code]
    select_plot_type = {
            "en-US": f'Select Plot Type',
            "pt-BR": f'Selecione o Tipo de Gráfico',
        }[lang_code]
    best_color_params_user_data = {
            "en-US": f'Best Color Parameters User Data',
            "pt-BR": f'Melhores Parâmetros de Cor por Usuário',
        }[lang_code]
    select_user = {
            "en-US": f'Select User',
            "pt-BR": f'Selecione o Usuário',
        }[lang_code]
    anonymous_user = {
            "en-US": lambda name: f'anonymous user {name}',
            "pt-BR": lambda name: f'usuário anônimo {name}',
        }[lang_code]
    priority_field_is_missing = {
            "en-US": f'Priority field is missing.',
            "pt-BR": f'O campo de prioridade está faltando.',
        }[lang_code]
    priority_field_description = {
            "en-US": f'User priority',
            "pt-BR": f'Prioridade do usuário',
        }[lang_code]
    processing = {
            "en-US": f'Processing...',
            "pt-BR": f'Processando...',
        }[lang_code]
    finished_processing = {
            "en-US": f'Finished processing.',
            "pt-BR": f'Processamento concluído.',
        }[lang_code]
    progress = {
            "en-US": 'Progress: {p}%',
            "pt-BR": 'Progresso: {p}%',
        }[lang_code]
    users_stats = {
            "en-US": 'Users Stats',
            "pt-BR": 'Estatísticas dos Usuários',
        }[lang_code]
    exit_session = {
            "en-US": 'You can exit the current session clicking the button below.',
            "pt-BR": 'Você pode sair da sessão atual clicando no botão abaixo.',
        }[lang_code]